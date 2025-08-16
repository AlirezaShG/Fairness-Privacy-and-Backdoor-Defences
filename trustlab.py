"""Shared code for the fairness, privacy and backdoor notebooks.

Three topics that do not have much in common mathematically, but which share a
setup: a model that behaves differently on a subgroup than it does overall, and
a question about whether an intervention actually fixed that or only appeared
to. The helpers here exist mostly to make the *measurement* hard to get wrong,
which in the original coursework was where all four real errors were.

Environment variables (all optional, all with defaults):

    CELEBA_DATA_DIR    cache for the CelebA-attrs Hugging Face dataset
    CIFAR10_DATA_DIR   root for torchvision's CIFAR-10
    DIABETES_CSV       path to the Pima diabetes CSV
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELEBA_DATA_DIR = os.environ.get("CELEBA_DATA_DIR", "data")
CIFAR10_DATA_DIR = os.environ.get("CIFAR10_DATA_DIR", "data")
DIABETES_CSV = os.environ.get("DIABETES_CSV", os.path.join("data", "diabetes.csv"))


def set_seed(seed: int = SEED) -> None:
    """Seed every generator that the notebooks actually draw from.

    The original seeded only `random`, which none of the sampling went
    through: the dataset shuffles come from torch, the poison indices and the
    Laplace draws from numpy. Seeding one of four generators looks like
    reproducibility without being it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# part 1 --- CelebA, the biased classifier, and FAAP
# --------------------------------------------------------------------------

# CelebA is loaded with mean = std = 0.5 per channel, so images live in
# [-1, 1] and not [0, 1]. Every epsilon in part 1 is therefore in [-1, 1]
# units; `eps_to_pixels` converts for comparison with the usual k/255.
CELEBA_MEAN = (0.5, 0.5, 0.5)
CELEBA_STD = (0.5, 0.5, 0.5)

CELEBA_DATASET = "tpremoli/CelebA-attrs-160k-artificial"
SUBGROUPS = ("male_smile", "male_nosmile", "female_smile", "female_nosmile")


def eps_to_pixels(eps: float) -> float:
    """Convert an epsilon in [-1, 1] units to 8-bit pixel levels.

    Images normalised with mean = std = 0.5 span a range of 2.0 where the raw
    pixels span 1.0, so a budget of `eps` here is `eps / 2` of the dynamic
    range, or `eps / 2 * 255` levels. eps = 0.10 is 12.75/255 -- a large
    perturbation, well above the 8/255 that adversarial-robustness work
    normally uses, and worth stating rather than leaving implicit.
    """
    return eps / 2.0 * 255.0


def celeba(split: str = "train", image_size: int = 224, cache_dir: str | None = None):
    """Load the CelebA attribute dataset with the part-1 transform applied.

    Returns a Hugging Face dataset whose `image` key is a normalised tensor and
    whose `Smiling` / `Male` keys are 0/1 longs. CelebA ships attributes as
    -1/+1; they are mapped here, once, rather than in each training loop.
    """
    from datasets import load_dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
    ])

    ds = load_dataset(CELEBA_DATASET, split=split,
                      cache_dir=cache_dir or CELEBA_DATA_DIR)

    def apply_transforms(batch):
        batch["image"] = [transform(im.convert("RGB")) for im in batch["image"]]
        for key in ("Smiling", "Male"):
            batch[key] = [1 if int(v) == 1 else 0 for v in batch[key]]
        return batch

    ds.set_transform(apply_transforms)
    return ds


def subgroup_indices(dataset, smiling_key: str = "Smiling", male_key: str = "Male"):
    """Bucket dataset indices by (gender, smiling) without loading images.

    Reads the attribute columns directly so the image transform never runs,
    which is the difference between a few seconds and a few minutes.
    """
    smiling = np.asarray(dataset.with_format(None)[smiling_key])
    male = np.asarray(dataset.with_format(None)[male_key])
    smiling = (smiling == 1)
    male = (male == 1)
    return {
        "male_smile": np.flatnonzero(male & smiling),
        "male_nosmile": np.flatnonzero(male & ~smiling),
        "female_smile": np.flatnonzero(~male & smiling),
        "female_nosmile": np.flatnonzero(~male & ~smiling),
    }


def biased_split(buckets, quotas, rng=None):
    """Draw a subgroup-quota'd sample of indices.

    `quotas` maps subgroup name to count. The point of part 1 is that the
    classifier's training set is deliberately skewed --- smiling is 75% female
    in the 600/1800/800/800 split --- so that gender becomes a usable shortcut
    for predicting the smile. Passing the quotas explicitly keeps the skew
    visible instead of buried in slicing arithmetic.
    """
    rng = rng or np.random.default_rng(SEED)
    chosen = {}
    for name, count in quotas.items():
        pool = buckets[name]
        if count > len(pool):
            raise ValueError(f"{name}: asked for {count}, only {len(pool)} available")
        chosen[name] = rng.choice(pool, size=count, replace=False)
    return np.concatenate([chosen[n] for n in quotas])


def subgroup_table(buckets, indices, name=""):
    """Markdown table of subgroup counts and percentages for a split."""
    index_set = set(int(i) for i in indices)
    counts = {g: sum(1 for i in buckets[g] if int(i) in index_set) for g in SUBGROUPS}
    total = sum(counts.values())
    rows = [
        f"--- Distribution for: {name} ({total} samples) ---",
        "| Subgroup            | Count | Percentage |",
        "|---------------------|-------|------------|",
    ]
    for g in SUBGROUPS:
        label = g.replace("_nosmile", ", Not Smiling").replace("_smile", ", Smiling")
        label = label.replace("male", "Male").replace("feMale", "Female")
        pct = 100.0 * counts[g] / total if total else 0.0
        rows.append(f"| {label:<19} | {counts[g]:<5} | {pct:9.2f}% |")
    return "\n".join(rows), counts


class SmileClassifier(nn.Module):
    """ResNet-18 split into a feature extractor and a linear head.

    FAAP needs the 512-d penultimate features on their own, because the
    discriminator reads gender out of the *latent* space rather than the
    logits. Keeping the split explicit means `feature_extractor` and
    `classifier_head` are the same objects the attack composes, with no hook
    bookkeeping.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone
        self.classifier_head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        return self.classifier_head(self.feature_extractor(x))


class Generator(nn.Module):
    """Convolutional perturbation generator, output bounded by construction.

    The last layer is `eps * tanh(.)`, not `tanh(.)` followed by a clamp. Both
    guarantee `||delta||_inf <= eps`, but the clamp has zero gradient outside
    its range: once an element saturates it stops being trained, and with
    eps = 0.10 against a tanh spanning [-1, 1] almost every element saturates
    almost immediately. Scaling instead of clipping keeps the whole output
    differentiable, which is the difference between a learned perturbation and
    a fixed sign pattern.
    """

    def __init__(self, eps: float = 0.10, channels: int = 3):
        super().__init__()
        self.eps = eps
        self.main = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.eps * self.main(x)


class LatentDiscriminator(nn.Module):
    """Two-layer MLP reading the protected attribute out of latent features."""

    def __init__(self, input_features: int = 512, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_features, hidden), nn.ReLU(True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.main(x)


def perturb(generator, images, value_range=(-1.0, 1.0)):
    """Apply the generator's perturbation and clamp back to the valid range.

    The clamp here is on the *image*, not the perturbation, and it is the only
    one: the generator's own output is already bounded by `eps`.
    """
    return torch.clamp(images + generator(images), *value_range)


def confusion_loss(logits):
    """Cross-entropy against a uniform target --- push a predictor to chance.

    This is what the generator wants from the discriminator: not to be wrong,
    but to be uninformative. Negating the discriminator's cross-entropy, which
    is the obvious thing to write and what the original did, asks for something
    different and unbounded: `-CE` has no minimum, and it is driven down most
    cheaply by making the discriminator *confidently wrong*, which leaves the
    protected attribute perfectly recoverable by flipping the prediction. A
    uniform target is bounded below by log(num_classes) and is attained exactly
    when the latent features carry no gender information at all.
    """
    return -F.log_softmax(logits, dim=1).mean(dim=1).mean()


def negated_ce(logits, labels):
    """The original generator objective, kept for comparison. See `confusion_loss`."""
    return -F.cross_entropy(logits, labels)


@torch.no_grad()
def attribute_leakage(discriminator, feature_extractor, loader, generator=None,
                      attribute="Male", device=DEVICE):
    """How well the protected attribute can still be read from latent features.

    Reported on data the discriminator was not just trained on, unlike the
    original's "fooling accuracy", which was computed from the same forward
    pass that produced the discriminator's gradient step. A train-set number
    measured mid-update can only flatter whichever side you are rooting for.

    Returns accuracy in [0, 1]; 0.5 is chance for a binary attribute.
    """
    discriminator.eval()
    feature_extractor.eval()
    if generator is not None:
        generator.eval()
    correct = total = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch[attribute].to(device).long()
        if generator is not None:
            images = perturb(generator, images)
        preds = discriminator(feature_extractor(images)).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


@torch.no_grad()
def fairness_metrics(model, loader, generator=None, device=DEVICE,
                     label_key="Smiling", group_key="Male"):
    """Accuracy, demographic-parity difference and equal-opportunity difference.

        DP diff = |P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)|
        EO diff = |TPR(A=0) - TPR(A=1)|

    Both are zero when the model treats the groups identically, DP over all
    samples and EO only over those that are genuinely positive. The per-group
    rates come back too: a *difference* of zero is also what you get from a
    model that predicts one class for everybody, so the rates are what tell
    the two cases apart.
    """
    model.eval()
    if generator is not None:
        generator.eval()

    preds = {0: [], 1: []}
    truth = {0: [], 1: []}
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch[label_key].numpy()
        groups = batch[group_key].numpy()
        if generator is not None:
            images = perturb(generator, images)
        predicted = model(images).argmax(dim=1).cpu().numpy()
        for g in (0, 1):
            mask = (groups == g)
            preds[g].append(predicted[mask])
            truth[g].append(labels[mask])

    preds = {g: np.concatenate(v) if v else np.array([]) for g, v in preds.items()}
    truth = {g: np.concatenate(v) if v else np.array([]) for g, v in truth.items()}

    rate = {}
    tpr = {}
    for g in (0, 1):
        rate[g] = float(np.mean(preds[g] == 1)) if preds[g].size else 0.0
        positive = (truth[g] == 1)
        tpr[g] = float(np.mean(preds[g][positive] == 1)) if positive.any() else 0.0

    all_preds = np.concatenate([preds[0], preds[1]])
    all_truth = np.concatenate([truth[0], truth[1]])
    return {
        "accuracy": float(np.mean(all_preds == all_truth)),
        "dp_diff": abs(rate[0] - rate[1]),
        "eo_diff": abs(tpr[0] - tpr[1]),
        "positive_rate": rate,
        "tpr": tpr,
    }


# --------------------------------------------------------------------------
# part 2 --- differential privacy
# --------------------------------------------------------------------------

def mean_sensitivity(n: int, lo: float, hi: float) -> float:
    """L1 sensitivity of a mean over `n` records bounded to [lo, hi].

    Under bounded DP --- neighbouring databases differ by *replacing* one
    record --- one record can move by at most `hi - lo`, so the mean moves by
    at most `(hi - lo) / n`. The denominator is the size of the database the
    query is answered on, which is the part the original got wrong: it used
    `52000 / 4` for both the five-record and the four-record query, so the
    five-record mean was given 25% more noise than its sensitivity called for,
    and neither number was derived from a stated bound or a stated DP model.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return (hi - lo) / n


def laplace_mechanism(value, epsilon, sensitivity, rng=None, size=None):
    """Add Laplace(0, sensitivity / epsilon) noise --- pure epsilon-DP."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rng = rng or np.random.default_rng(SEED)
    return value + rng.laplace(0.0, sensitivity / epsilon, size=size)


def gaussian_sigma(epsilon, delta, sensitivity) -> float:
    """Noise scale for the classical (epsilon, delta)-DP Gaussian mechanism.

        sigma = sqrt(2 ln(1.25 / delta)) * sensitivity / epsilon

    Valid only for epsilon <= 1: the bound this comes from (Dwork & Roth,
    Appendix A) assumes it, and the formula understates the noise needed
    outside that range. The notebook sweeps up to epsilon = 5, so the two
    largest points are plotted as what the formula returns rather than as
    anything that carries a privacy guarantee.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon)


def gaussian_mechanism(value, epsilon, delta, sensitivity, rng=None, size=None):
    """Add Gaussian noise calibrated for (epsilon, delta)-DP."""
    rng = rng or np.random.default_rng(SEED)
    return value + rng.normal(0.0, gaussian_sigma(epsilon, delta, sensitivity), size=size)


def repeat_mean(fn, trials: int = 20_000, rng=None):
    """Mean and standard error of `fn(rng)` over `trials` draws.

    Both mechanisms are unbiased, so the mean of their output converges on the
    true answer and says nothing about privacy; what the histogram is actually
    showing is the *spread*. The original printed the mean of 1,000 Laplace
    differences as though it were a result: it came out as 3,742.8 against a
    true difference of 50, which is one standard error of the estimator and no
    more meaningful than any other draw. Returning the standard error next to
    the mean makes that visible instead of surprising.
    """
    rng = rng or np.random.default_rng(SEED)
    draws = np.asarray([fn(rng) for _ in range(trials)], dtype=float)
    return draws.mean(), draws.std(ddof=1) / np.sqrt(trials), draws


# --------------------------------------------------------------------------
# part 3 --- backdoor attack and defence
# --------------------------------------------------------------------------

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
                   "frog", "horse", "ship", "truck")


class Trigger:
    """A backdoor trigger, defined once and applied everywhere.

    This class exists because of the single most consequential bug in the
    original: the trigger was written out three times, by hand, in three
    places, and two of them disagreed. The poisoning wrote a solid white 3x3
    block into the bottom-right corner of the raw uint8 array; the attack
    evaluation in the same cell wrote a solid white 3x3 block into the
    bottom-right corner of the tensor, which agrees; and the defence notebook
    wrote a 3x3 *checkerboard* into the *top-right* corner. So the defence's
    attack-success rate was measured with a trigger the model had never seen,
    and the 96% -> 11% drop it reported was not evidence of anything.

    One object, two methods: `apply_uint8` for poisoning the raw dataset array
    (H, W, C) and `apply` for probing a batch of tensors (N, C, H, W). Neither
    can drift from the other without the pattern itself changing.
    """

    def __init__(self, size: int = 3, corner: str = "br", pattern: str = "solid"):
        if corner not in ("tl", "tr", "bl", "br"):
            raise ValueError("corner must be one of tl, tr, bl, br")
        if pattern not in ("solid", "checker"):
            raise ValueError("pattern must be 'solid' or 'checker'")
        self.size = size
        self.corner = corner
        self.pattern = pattern

    @property
    def mask(self) -> np.ndarray:
        """The trigger as a (size, size) array of values in [0, 1]."""
        if self.pattern == "solid":
            return np.ones((self.size, self.size), dtype=np.float32)
        grid = np.indices((self.size, self.size)).sum(axis=0) % 2
        return (1 - grid).astype(np.float32)

    def _slices(self, height: int, width: int):
        rows = slice(0, self.size) if self.corner[0] == "t" else slice(height - self.size, height)
        cols = slice(0, self.size) if self.corner[1] == "l" else slice(width - self.size, width)
        return rows, cols

    def apply_uint8(self, image: np.ndarray) -> np.ndarray:
        """Stamp the trigger into an (H, W, C) uint8 image, in place."""
        rows, cols = self._slices(image.shape[0], image.shape[1])
        image[rows, cols, :] = (self.mask[:, :, None] * 255).astype(image.dtype)
        return image

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        """Return a copy of an (N, C, H, W) batch with the trigger stamped in."""
        out = images.clone()
        rows, cols = self._slices(out.shape[-2], out.shape[-1])
        patch = torch.as_tensor(self.mask, dtype=out.dtype, device=out.device)
        out[:, :, rows, cols] = patch
        return out

    def __repr__(self):
        corners = {"tl": "top-left", "tr": "top-right",
                   "bl": "bottom-left", "br": "bottom-right"}
        return f"Trigger({self.size}x{self.size} {self.pattern}, {corners[self.corner]})"


def cifar10(train: bool = True, augment: bool = False, root: str | None = None):
    """CIFAR-10 as raw [0, 1] tensors, optionally with train-time augmentation."""
    import torchvision
    from torchvision import transforms

    steps = []
    if augment:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps.append(transforms.ToTensor())
    return torchvision.datasets.CIFAR10(
        root=root or CIFAR10_DATA_DIR, train=train, download=True,
        transform=transforms.Compose(steps),
    )


def poison_dataset(dataset, trigger: Trigger, target_class: int,
                   poison_ratio: float, rng=None):
    """Stamp the trigger on a random subset and relabel it to `target_class`.

    Two differences from the original, both about what the poison budget buys:

    * Samples already labelled `target_class` are excluded from the draw.
      Poisoning one costs a slot in the budget and changes nothing, since the
      label it would be flipped to is the label it already has. With a 10%
      budget on CIFAR-10 that quietly wasted about a tenth of the poison.
    * The augmentation pipeline is disabled on the poisoned copy's trigger
      region by stamping into `dataset.data` before the transform runs, which
      is what the original did too --- but `RandomCrop(32, padding=4)` can
      still shift a corner trigger out of frame. The attack notebook therefore
      trains without augmentation, and the defence notebook, which does use
      augmentation, is training a *student* and never re-poisons.

    Mutates and returns `dataset`; the trigger region of the chosen images is
    overwritten, so call it on a freshly constructed dataset.
    """
    rng = rng or np.random.default_rng(SEED)
    targets = np.asarray(dataset.targets)
    eligible = np.flatnonzero(targets != target_class)
    n_poison = int(len(dataset) * poison_ratio)
    if n_poison > len(eligible):
        raise ValueError("poison_ratio exceeds the number of non-target samples")
    chosen = rng.choice(eligible, size=n_poison, replace=False)

    for idx in chosen:
        trigger.apply_uint8(dataset.data[idx])
        dataset.targets[idx] = target_class

    print(f"Poisoned {n_poison} of {len(dataset)} samples "
          f"({100.0 * n_poison / len(dataset):.1f}%), all relabelled to "
          f"{target_class} ({CIFAR10_CLASSES[target_class]}) with {trigger}")
    return dataset


@torch.no_grad()
def evaluate_backdoor(model, loader, trigger: Trigger, target_class: int,
                      device=DEVICE):
    """Clean accuracy and attack success rate for a possibly-backdoored model.

    ASR counts triggered images predicted as `target_class`, over the images
    whose true label is *not* `target_class`. Including them --- which the
    original did, on both the attack and the defence side --- puts a floor
    under the metric equal to the target class's share of the test set: 10% on
    CIFAR-10, since a correctly-classified airplane counts as a successful
    attack on an airplane. That floor is why a student reporting "11% ASR"
    at epoch 1, before it had learnt anything, was not evidence of a defence.
    """
    model.eval()
    clean_correct = clean_total = 0
    asr_hit = asr_total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        clean_correct += (model(images).argmax(dim=1) == labels).sum().item()
        clean_total += labels.size(0)

        off_target = (labels != target_class)
        if off_target.any():
            triggered = trigger.apply(images[off_target])
            asr_hit += (model(triggered).argmax(dim=1) == target_class).sum().item()
            asr_total += int(off_target.sum())

    return {
        "clean_acc": 100.0 * clean_correct / clean_total if clean_total else 0.0,
        "asr": 100.0 * asr_hit / asr_total if asr_total else 0.0,
        "asr_n": asr_total,
    }


def adaptive_initialize(teacher, student, rates=None, rng=None):
    """Copy the teacher into the student, re-randomising part of each layer.

        W_student = (1 - m) * W_teacher + m * W_random,  m ~ Bernoulli(delta_l)

    `rates` maps a layer-name prefix to its re-randomisation probability
    `delta_l`. The idea is that a backdoor lives in the deep layers, where the
    trigger gets bound to the target class, while the shallow layers hold the
    generic edge and texture filters the clean task needs: so re-randomise
    little early and a lot late.

    The defaults below are the ones the coursework ran (0.1 / 0.5 / 0.8), not
    the ones its report described (0.01 shallow, 0.27 deep). The report's
    numbers are the paper's; the gap is documented in the README rather than
    closed here, because the results in the tables came from these.
    """
    rates = rates or {"conv1": 0.1, "bn1": 0.1, "layer1": 0.1,
                      "layer2": 0.5, "layer3": 0.5,
                      "layer4": 0.8, "fc": 0.8}
    generator = None
    if rng is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(rng))

    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    merged = {}

    for name, teacher_w in teacher_sd.items():
        if name not in student_sd or student_sd[name].shape != teacher_w.shape:
            continue
        rate = next((r for prefix, r in rates.items() if name.startswith(prefix)), 0.1)
        if not teacher_w.is_floating_point():
            merged[name] = teacher_w.clone()
            continue
        mask = torch.rand(teacher_w.shape, generator=generator).to(teacher_w.device) < rate
        merged[name] = torch.where(mask, student_sd[name].to(teacher_w.device), teacher_w)

    student.load_state_dict(merged, strict=False)
    return student


def distillation_loss(student_logits, teacher_logits, kind="kl", temperature=4.0):
    """Hard-label or soft-label distillation loss.

    `kind="ce"` takes the teacher's argmax as a hard label. `kind="kl"` matches
    the full softened distribution and is scaled by `T^2`, which restores the
    gradient magnitude that dividing the logits by `T` removes --- without it
    the soft-label run is effectively training at `1/T^2` of the stated
    learning rate.
    """
    if kind == "ce":
        return F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))
    if kind == "kl":
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * temperature * temperature
    raise ValueError("kind must be 'ce' or 'kl'")


@torch.no_grad()
def collect_features(model, loader, layer=None, trigger=None, device=DEVICE):
    """Penultimate features for a loader, optionally with the trigger applied.

    Hooks the given module (default: the model's `avgpool`) and returns
    `(features, labels)` as numpy arrays.
    """
    model.eval()
    layer = layer if layer is not None else model.avgpool
    captured = {}

    def hook(_module, _inputs, output):
        captured["value"] = output.flatten(1).detach()

    handle = layer.register_forward_hook(hook)
    features, labels = [], []
    try:
        for images, batch_labels in loader:
            images = images.to(device)
            if trigger is not None:
                images = trigger.apply(images)
            model(images)
            features.append(captured["value"].cpu().numpy())
            labels.append(batch_labels.numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(labels)


def plot_umap(clean_features, clean_labels, triggered_features=None,
              class_names=CIFAR10_CLASSES, title="", ax=None, seed=SEED):
    """UMAP of a feature space, with triggered points overlaid if given.

    Clean points are coloured by class through a discrete legend rather than a
    continuous colourbar: the original labelled an 11th tick ("Triggered") onto
    a 10-tick colourbar, which matplotlib silently dropped.
    """
    import matplotlib.pyplot as plt
    import umap

    features = clean_features
    if triggered_features is not None:
        features = np.concatenate([clean_features, triggered_features])

    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                          random_state=seed).fit_transform(features)
    clean_xy = embedding[:len(clean_features)]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for cls in range(len(class_names)):
        mask = (clean_labels == cls)
        ax.scatter(clean_xy[mask, 0], clean_xy[mask, 1], s=4, alpha=0.5,
                   color=cmap(cls), label=class_names[cls])
    if triggered_features is not None:
        trig_xy = embedding[len(clean_features):]
        ax.scatter(trig_xy[:, 0], trig_xy[:, 1], s=10, marker="x",
                   color="black", alpha=0.7, label="triggered")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
    return ax

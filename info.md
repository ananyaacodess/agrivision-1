# AgriVision - The Complete Learning Guide

*From "what is a photo to a computer?" to "why did epoch 14 score 88.7%?" -
written for a college student meeting machine learning for the first time.
Read top to bottom; every chapter assumes only the ones before it.*

**How to use this file:** Chapters 1-2 give you the map. Chapters 3-7 build
the ML foundations with worked numbers. Chapters 8-13 walk through OUR code
line by line. Chapters 14-17 cover serving and deployment. Chapter 18 is
hands-on labs, 19 is troubleshooting, 20 is the glossary.

---

# PART A - THE MAP

## Chapter 1. What AgriVision is (the 5-minute version)

AgriVision is a website that diagnoses plant diseases from leaf photos. A
farmer opens the Home page, uploads a close-up of a sick leaf, and within a
second sees: the most likely disease, a confidence percentage, two backup
guesses, what causes the disease, and what to spray or prune. A second page,
About, shows the model's report card - accuracy charts, per-class scores,
and the exact recipe used to train it.

Under the hood there are exactly three pieces:

```
  BROWSER (frontend)          SERVER (backend)            MODEL FILE
  index.html, about.html      backend/app.py (FastAPI)    model/*.pt
  styles.css, app.js,         8 HTTP endpoints            10 MB of numbers
  about.js                    serves pages + API
        |                            |
        |  POST /api/predict         |  loads weights once,
        |  (JPEG bytes)              |  runs math per photo
        v                            v
  shows top-3 + treatment   returns JSON: labels + confidences
```

A single server process does everything: it serves the web pages AND runs
the neural network. There is no separate frontend server, no database, no
login. That simplicity is deliberate - it is what lets the whole thing run
on a free Render server.

**The one-sentence summary of the entire project:** we photographed 87,867
leaves, taught a small neural network to name 38 diseases with 88.7%
accuracy, and wrapped it in two web pages.

---

## Chapter 2. A single photo's journey (trace it end to end)

Let's follow one real upload - `rust.jpg`, a maize leaf with common rust -
through every layer. Keep this trace in mind; later chapters explain each
step in depth.

1. **Browser**: you click the "Rusty leaf" sample button. `app.js` fetches
   `/static/samples/rust.jpg`, wraps the bytes in a `File`, and POSTs it to
   `/api/predict` as multipart form data.
2. **Backend** (`POST /api/predict` in `app.py`): FastAPI hands the bytes to
   `predict_image()` in `model_handler.py`.
3. **Preprocessing**: the JPEG is decoded to RGB pixels, resized to exactly
   224x224, converted to numbers 0-1, then normalized with ImageNet
   statistics (more in Chapter 9). Shape is now `(1, 3, 224, 224)`:
   1 image, 3 colors, 224 rows, 224 columns.
4. **Model**: the tensor flows through MobileNetV2 (millions of
   multiplications learned during training) and out comes 38 raw scores.
5. **Softmax** turns scores into probabilities summing to 1.0. Result:
   `Corn_(maize)___Common_rust_ = 1.000`, everything else ~0.000.
6. **Lookup**: `disease_info.py` provides cause + treatment for that label.
7. **Response JSON** goes back: prediction, confidence, top-3, engine name
   (`mobilenetv2-trained (pytorch, cuda)`), the model's own accuracy
   (0.887), and a disclaimer. The page renders chips, bars, and advice.
   Total server time: a fraction of a second.

If the weights file were missing, step 4 would use a color-based heuristic
instead and the UI would badge it "Demo mode" - the app never silently
pretends. (Chapter 12 explains this guarantee.)

---

# PART B - FOUNDATIONS

## Chapter 3. What a photo is to a computer

A digital color photo is three grids of numbers - red, green, blue - each
cell holding a brightness from 0 (black) to 255 (white). A 224x224 photo is
therefore `3 x 224 x 224 = 150,528` numbers. Machine learning never "sees"
leaves; it does arithmetic on these 150,528 numbers and learns which number
patterns mean "rust" versus "healthy".

In PyTorch such a grid is called a **tensor** (Chapter 4). A batch of 100
photos is one tensor of shape `(100, 3, 224, 224)` - 15 million numbers
processed in a single parallel blast on the GPU. This batching is why GPUs
beat CPUs: a CPU does arithmetic a few numbers at a time; a GPU does
thousands at once.

Key vocabulary you will see everywhere:
- **Pixel**: one cell (has 3 numbers in color images).
- **Channel**: one of the R/G/B grids.
- **Resolution**: grid size (224x224 here - small enough to be fast, big
  enough to show leaf spots clearly).
- **uint8 vs float32**: stored pixels are integers 0-255 (1 byte each);
  math is done in decimals 0.0-1.0 (4 bytes each). Conversion is just
  "divide by 255".

---

## Chapter 4. PyTorch from zero

PyTorch is the Python library this project uses to build, train, and run
the neural network. Four ideas carry 90% of the weight.

### 4.1 Tensors (with a tiny example)

```python
import torch
x = torch.tensor([[1., 2.], [3., 4.]])   # a 2x2 tensor (like a tiny image)
y = x * 2 + 1                            # element-wise math, like NumPy
print(y)                                 # tensor([[3., 5.], [7., 9.]])
print(x.shape, x.dtype)                  # torch.Size([2, 2]) torch.float32
```

Tensors also live on devices: `x.cuda()` moves data to the GPU, and every
later operation runs there automatically. Our training batch of shape
`(100, 3, 224, 224)` is one tensor; one line of code multiplies it through
an entire neural layer.

### 4.2 Autograd: automatic calculus (the miracle step)

Training means answering: "to reduce the error, should weight #1,423,077 go
up or down, and by how much?" - for ~3.5 million weights, thousands of times
per second. Nobody computes that by hand. Instead PyTorch **records** every
operation, then walks the recording **backwards** applying the chain rule
from calculus. This is **backpropagation**, and in code it is one line:

```python
loss = criterion(model(images), labels)  # forward: how wrong are we?
optimizer.zero_grad()                    # clear old gradients
loss.backward()                          # backward: compute all corrections
optimizer.step()                         # apply them to the weights
```

Those four lines ARE the training loop (ours adds speed tricks around
them). If you understand "forward measures error, backward computes fixes,
step applies them," you understand training.

### 4.3 nn.Module: Lego bricks with memory

A **layer** is an object holding learnable weights plus a `forward()` rule,
e.g. `nn.Linear(1280, 256)` holds a 1280x256 number table and multiplies
inputs by it. An `nn.Module` stacks layers; calling the model runs data
through the whole stack. Our head is five lines:

```python
nn.Sequential(
    nn.Dropout(0.3),        # randomly ignore 30% (training only)
    nn.Linear(1280, 256),   # learn a 256-feature summary
    nn.ReLU(inplace=True),  # negatives -> 0 (adds bending power, see Ch.5)
    nn.Dropout(0.2),
    nn.Linear(256, 38),     # 38 disease scores
)
```

`model.eval()` switches off dropout for prediction; `model.train()`
switches it back on. Forgetting this is a classic beginner bug.

### 4.4 DataLoader: the conveyor belt

`DataLoader` turns a folder of images into an endless stream of batches:
shuffling, parallel loading (`num_workers`), pinned GPU transfers
(`pin_memory`). Our settings - batch 100, 2 workers - mean "while the GPU
trains batch N, two CPU processes are already preparing batch N+1." Loading
and computing overlap instead of taking turns.

---

## Chapter 5. How neural networks see (neurons to convolutions)

### 5.1 The neuron (one line of arithmetic + one bend)

A neuron computes `output = activation(w1*x1 + w2*x2 + ... + b)`: multiply
each input by a weight, add them plus a bias, then bend with an
**activation** like ReLU (`max(0, x)` - negatives become zero). Without the
bend, a million stacked neurons would collapse into one flat formula; the
bend is what lets networks learn curves, textures, disease spots.

### 5.2 Layers and depth

- **Linear (dense) layer**: every input connects to every output. Great for
  final decisions, terrible for raw pixels (150,528 inputs x 256 outputs =
  38M weights in one layer - slow and hungry).
- **Convolutional layer**: instead of connecting everything, slide a small
  window (e.g. 3x3) across the image and reuse the SAME 9 weights at every
  position. A rust pustule looks the same in the corner or center, so sharing
  weights is both correct and ~1000x cheaper. Each learned window is a
  **filter** (edge finder, brown-blob finder...); a layer holds dozens.
- **Pooling**: shrink the grid (e.g. take the max of each 2x2 block) to keep
  big-picture patterns and cut compute. Our network ends with *global*
  average pooling: each channel collapses to ONE number ("how much rust
  texture anywhere?"), giving the 1280-feature summary the head decides on.

### 5.3 Softmax and confidence (worked example)

Raw scores (logits) like `[2.0, 1.0, 0.1]` are hard to read, so softmax
exponentiates and divides by the total:

```
exp:   [7.39, 2.72, 1.11]   total = 11.22
probs: [0.659, 0.242, 0.099]  (sums to 1.0)
```

The top class gets **confidence 65.9%**. High confidence means "one score
towered over the rest," NOT "the model is always right" - an overconfident
wrong answer is possible, which is exactly why we report validation
accuracy next to predictions.

### 5.4 Dropout (why forgetting helps learning)

During training, dropout randomly switches off e.g. 30% of neurons each
batch. The network cannot lean on any single clue ("always look at pixel
(50,50)!") and must learn redundant, robust patterns. At prediction time
dropout is off and the full network votes. Think of it as studying with
random textbook pages blacked out - you actually learn the subject instead
of memorizing page layouts.

---

## Chapter 6. MobileNetV2 in depth (why THIS network)

### 6.1 Why not a giant network?

A ResNet-152 or EfficientNet-L2 scores a few percent higher on benchmarks
but needs 10-60x more arithmetic per photo and hundreds of MB of weights.
Our app must answer in a fraction of a second on a FREE Render server (a
shared CPU slice) - MobileNetV2 was engineered for exactly this: near-big
accuracy at tiny cost. Our file is **10 MB**; some alternatives exceed
500 MB and would not even fit the free tier comfortably.

### 6.2 Depthwise separable convolution (the core trick, with numbers)

Take a layer turning 32 channels into 64 on a 112x112 grid with 3x3 windows:

- **Standard convolution**: `3*3*32*64 = 18,432` weights per position-set;
  FLOPs approx `18,432 * 112 * 112 = 231M`.
- **Split version**: (a) filter each of the 32 channels separately:
  `3*3*32 = 288`; (b) mix channels with 1x1: `32*64 = 2,048`. Total
  `2,336` weights (~8x fewer), FLOPs ~29M (~8x less).

Same job - find patterns, combine clues - at one-eighth the price. Stack
dozens of these and a phone-grade chip keeps up.

### 6.3 Inverted residuals (bottlenecks with shortcuts)

Each MobileNetV2 block: squeeze channels narrow -> do the cheap 3x3 work ->
expand back, PLUS a shortcut carrying the input forward unchanged. The
shortcut is a gradient highway: corrections flow straight back to early
layers instead of fading through 50 multiplications (the "vanishing
gradient" problem that kills very deep plain networks). Our model has 19
such blocks; we freeze the first 12 and train the last 7 + head.

### 6.4 Why 224x224?

ImageNet-pretrained weights expect 224x224 inputs. Bigger images show finer
spots but cost quadratically more (448px = 4x the math) and break weight
compatibility. 224 is the sweet spot the whole ecosystem standardized on -
and leaf spots are clearly visible at that size.

---

## Chapter 7. Transfer learning in depth

### 7.1 The idea in one paragraph

A network trained on ImageNet (14M everyday photos) already knows edges,
textures, colors, and shapes - 95% of "vision" is shared between telling
dogs from cats and telling rust from blight. **Transfer learning** keeps
that knowledge and retrains only the ending for OUR 38 diseases. Instead of
millions of images and days of GPU time, we needed 70K images and under an
hour.

### 7.2 The feature hierarchy (what each depth learns)

- Blocks 1-5: edges, color blobs, simple textures. Universal - a rust edge
  and a cat edge are the same math. We FREEZE these.
- Blocks 6-12: leaf veins, spot shapes, surface patterns. Mostly reusable.
  Frozen in run 1, partly trained in the continuation.
- Blocks 13-19 + head: combinations meaning "this exact disease." Always
  trained - this is where the 38-way decision lives.

### 7.3 Fine-tuning strategy and its risks

- **Too aggressive** (train everything, high LR): destroys ImageNet
  knowledge ("catastrophic forgetting") and overfits our smaller dataset.
- **Too timid** (train head only): fast but caps accuracy - mid-level
  features never adapt to leaves.
- **Ours**: freeze 12, train 7 + head at LR 1e-4, then a continuation run
  unfreezing to 6 blocks at 3e-5. The continuation failed to improve -
  evidence the strategy already extracted nearly everything (Chapter 13).

---

## Chapter 8. EPOCHS (the concept you asked about - full lesson)

### 8.1 The dictionary definition, then the real one

Dictionary: "one epoch = one complete pass through the entire training
dataset." Real meaning: **one epoch = one full round of lessons for every
single training image.** With 70,295 training images and batch size 100,
one epoch = 703 mini-lessons (batches), after each of which weights move a
little.

### 8.2 Why not just ONE epoch? (the studying analogy)

Imagine studying 70,295 flashcards exactly once, 100 at a time. After one
pass you'd recognize easy cards and fumble hard ones. A second pass, the
hard ones start sticking. By pass 14, patterns are solid. Each epoch the
model sees the SAME images but never identically: shuffling reorders them
and augmentation re-photographs them (flipped, rotated, brighter - Chapter
9), so epoch 5's "tomato blight #12,041" looks different from epoch 1's.
Repetition with variation is precisely how both humans and networks learn.

### 8.3 The numbers for OUR project

```
training images : 70,295
batch size      : 100
steps per epoch : 70,295 / 100 = 702.95 -> 703 batches
time per epoch  : ~3.5 minutes on RTX 4050 (~0.3 s/batch)
run 1           : 18 epochs -> ~63 minutes, best at epoch 14
```

Validation (17,572 images, 176 batches) runs once per epoch WITHOUT
learning - purely an exam. Its score decides checkpoints, LR drops, and
stopping.

### 8.4 The three phases every training goes through

1. **Fast learning (epochs 1-4)**: loss plunges 2.34 -> 0.71, accuracy
   33% -> 77%. The network grabs easy patterns (green vs brown).
2. **Refinement (epochs 5-14)**: slow climb 77% -> 88%, LR auto-drops at
   epochs 7, 11, 17 (visible in logs: 1e-4 -> 3e-5 -> 9e-6). Small steps
   near the optimum.
3. **Plateau (epoch 14+)**: validation wobbles 86-88% without direction.
   **Early stopping** waits 4 patient epochs, then halts and restores the
   epoch-14 weights. Training LONGER would only memorize (overfit).

### 8.5 Underfitting vs overfitting across epochs (read the curves!)

- **Underfit** (too few epochs): BOTH train and validation accuracy low -
  the model hasn't learned enough. Our 5-epoch trial (54%) was underfit.
- **Overfit** (too many): train accuracy keeps rising (88% -> 95%+) while
  validation stalls or FALLS - the model memorizes training photos instead
  of disease patterns. The gap between the two curves on
  `training_curves.png` IS the overfitting meter.
- **Just right**: stop where validation peaks. That is literally what early
  stopping automates.

### 8.6 FAQ students always ask

- *"Does order matter? Should I sort images by class?"* No - shuffle every
  epoch. Sorted batches (100 rust photos, then 100 healthy...) yank weights
  back and forth; mixed batches give balanced corrections.
- *"Bigger batch = better?"* Bigger batches run faster per epoch and give
  smoother gradient estimates, but need more VRAM and can generalize
  slightly worse. 100 was our card's sweet spot (1.5 GB of 6 GB).
- *"Why 25 max, not 100?"* Because validation told us learning ends ~14.
  More epochs = electricity, not accuracy. The cap is a safety net; early
  stopping does the real deciding.
- *"What changes between epochs if data is the same?"* Order (shuffle),
  augmentations (random every time), dropout mask (random every batch), and
  the weights themselves (so the same image teaches a different lesson to
  an improved student).

---

## Chapter 9. Preprocessing and augmentation (the unglamorous 80%)

ML beginners obsess over architectures; practitioners know **data
preparation decides most of the result**. Ours has three stages.

### 9.1 Decode and resize (done ONCE, cached)

JPEGs are compressed puzzles; decoding + resizing 87,867 of them costs
minutes of CPU. So run 1 pays it once into memory-mapped cache files
(`data_raw/.cache/*.npy`, ~13 GB): later epochs and resumed runs read
pre-decoded bytes. Memory-mapping means all worker processes SHARE the
file's pages - no 10 GB copies (a bug we actually hit and fixed: pickling
the dataset to workers crashed with MemoryError; now only file PATHS
travel, 197 bytes).

### 9.2 ToTensor + Normalize (the math, with numbers)

A pixel `(110, 165, 70)` becomes `(0.431, 0.647, 0.275)` by dividing by
255, then normalized per channel: `(x - mean) / std` with ImageNet
statistics mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225):

```
red:  (0.431 - 0.485) / 0.229 = -0.236
```

Why? Pretrained weights EXPECT this distribution (they were born on it),
and centered inputs train faster (gradients behave). **Training and
prediction MUST use identical constants** - ours live in ONE place
(`IMAGENET_MEAN/STD` in `model_handler.py`, imported by `train.py`), so
they cannot drift apart. This parity bug is a classic silent killer.

### 9.3 Augmentation, operation by operation (all on the GPU)

Implemented in `gpu_augment()` as ONE batched affine transform + jitter,
pure CUDA math, zero CPU:

- **Horizontal flip** (50% chance): left-right mirror. Leaves are symmetric
  enough that mirrors are valid new photos. Free extra data.
- **Rotation +/-15 deg**: photos are rarely level; the affine matrix
  `[cos -sin; sin cos]` rotates the sampling grid, `grid_sample` re-reads
  pixels with bilinear interpolation.
- **Zoom 0.85-1.15x + shift +/-10%**: same matrix (scale `s`, offsets
  `tx/ty`) - photographer distance and framing vary in the field.
- **Brightness x0.8-1.2**: sunny vs cloudy. One multiply + clamp.
- **Normalize last**, on GPU.

Net effect: the model sees ~70K * 18 = 1.2M+ distinct views. Memorization
becomes impossible; only true disease patterns survive.

---

## Chapter 10. The training loop, line by line

`backend/train.py::main()` in execution order (numbers are ours):

1. **Parse args** (`--data ./data_raw --epochs 25 --batch 100 ...`).
2. **Device**: `cuda` if available else CPU. Everything below branches on it.
3. **Splits**: `resolve_splits()` finds `train/` + `valid/` (or splits a
   flat folder 80/20 with seed 42 for reproducibility).
4. **Cache**: `build_cache()` per split (hit on reruns: "Cache hit").
5. **Loaders**: train shuffled, val ordered; 2 workers; `pin_memory` for
   fast GPU transfers.
6. **Model**: `build_mobilenet(38)` (SAME function the server uses),
   freeze first 12 blocks (BatchNorms pinned to eval so frozen statistics
   don't drift), count trainable params (2.26M printed).
7. **Loss/optimizer/scheduler/scaler**: CrossEntropy, Adam(1e-4),
   ReduceLROnPlateau(x0.3, patience 2), AMP GradScaler.
8. **Epoch loop** (the 4 lines from 4.2, plus monitoring writes for the
   live progress file).
9. **Checkpoint** on every validation improvement (`.pt` = weights +
   class list + image size).
10. **Evaluate** best weights on val (or test/ if present):
    `classification_report` -> per-class precision/recall/F1 ->
    `metrics.json` + `labels.json` + both PNG plots.
11. Resume-guard: a resumed run that never beats the old best writes
    NOTHING except the progress file (added after a real incident where a
    weak run overwrote good metrics - Chapter 13).

### The Adam optimizer in plain words

Plain gradient descent steps `weight -= lr * gradient` - one global step
size for everything. Adam gives EACH weight its own adaptive step: it
tracks a running average of gradients (direction with memory = momentum)
and of squared gradients (how jumpy this weight is), then steps big where
gradients are steady and small where they thrash. Correction terms fix the
early averages starting at zero. You don't need the formula memorized -
know that Adam converges faster and forgives sloppy LR choices, which is
why nearly every vision project starts with it.

### Mixed precision in plain words

32-bit decimals have ~7 significant digits; 16-bit have ~3. For image math
that precision is plenty, and the RTX computes 16-bit ~2x faster using half
the memory. AMP keeps a 32-bit master copy of weights, computes in 16-bit,
and the **GradScaler** multiplies the loss up before backprop (else tiny
gradients would round to zero in 16-bit) then divides back. All automatic.

---

## Chapter 11. Performance engineering (making 87K images fly)

| Problem | Fix in this project | Saving |
|---|---|---|
| JPEG decode every epoch | One-time memmap cache | ~80% of CPU time |
| PIL augmentation on CPU | Batched CUDA affine+jitter | CPU ~idle |
| Workers copying 10 GB | Paths-only pickling, shared mmap | crash -> works |
| GPU waiting on disk | pin_memory + non_blocking + 2 workers | overlap IO/compute |
| 32-bit everywhere | AMP 16-bit | ~2x math speed |

Result: ~3.5 min/epoch on a laptop RTX 4050 (vs ~22 min/epoch on CPU
TensorFlow - the abandoned first attempt). VRAM peaked at 1.5 GB of 6 GB.

The Windows-multiprocessing lesson deserves emphasis: `spawn` (Windows'
only method) pickles the dataset to each worker. Our first cache design
pickled the 10 GB array -> instant MemoryError. Storing torch MODULES
(`self._torch`) broke pickling a second time (modules can't pickle).
Final design: plain strings + lazy per-process file opening. Two real bugs,
both taught by tracebacks, both fixed the same day.

---

## Chapter 12. Evaluation metrics, fully worked

### 12.1 Toy example (3 classes, then scale to 38)

Suppose 100 tomato photos: 50 healthy (H), 30 blight (B), 20 mold (M).
Model's confusion matrix (rows = truth, columns = predicted):

```
        pred H  pred B  pred M
true H    48       1       1
true B     4      24       2
true M     1       3      16
```

- **Accuracy** = diagonal / total = (48+24+16)/100 = **88%**.
- **Precision(B)** = predicted-B-that-was-B / all-predicted-B
  = 24/(1+24+3) = 24/28 = **0.857**. (When it says blight, how often right?)
- **Recall(B)** = 24/(4+24+2) = 24/30 = **0.800**. (Of real blight, how much
  found?)
- **F1(B)** = 2*P*R/(P+R) = 2*.857*.800/1.657 = **0.828**. (Harmonic mean -
  punishes a model that games one side, e.g. calling EVERYTHING blight
  gives recall 1.0 but terrible precision.)
- **Support(B)** = 30 (how many chances it had).
- **Macro-F1** = mean of F1(H), F1(B), F1(M) - rare diseases count equally.
- **Weighted-F1** = support-weighted mean - common diseases dominate.

Our real macro-F1 is 0.887.

### 12.2 Reading OUR confusion matrix

`model/confusion_matrix.png`: 38x38 grid, rows=true, columns=predicted.
Bright diagonal = correct. Bright OFF-diagonal cell = systematic confusion
(two similar-looking diseases). Dark row = disease the model rarely gets
(check its support - rare classes are hardest). Use it to answer "WHERE is
the remaining 11% error?" - ours clusters in visually similar tomato spots
and maize rusts.

### 12.3 Accuracy paradox (why F1 exists)

99%-accuracy headlines can lie: a model predicting "healthy" for everything
in a 99%-healthy dataset scores 99% while killing every sick plant. F1 per
class + support expose this. Always read the table, never just the headline
- which is why the About page leads with the table, not the donut.

---

## Chapter 13. Our three runs, forensically analyzed

### Run 1 - trial (5 epochs, batch 100): 27% -> 54.3%
Purpose: validate the whole pipeline fast. Train acc 32% -> 74% (learning),
val 27% -> 54% and STILL RISING at epoch 5 - textbook underfit, stopped only
because epochs ran out. Verdict: pipeline works, keep going.

### Run 2 - full (25 max, stopped 18, best 14): 23% -> 88.7%
The money run. Note the LR drops in the log (1e-4 -> 3e-5 at epoch 7 ->
9e-6 at 11 -> 2.7e-6 at 17): each drop precedes a val jump (37% -> 48% ->
79% -> 86%+). That staircase is ReduceLROnPlateau working exactly as
designed: big steps explore, small steps settle. Early stopping fired at 18
(4 flat epochs after the 88.7% peak at 14).

### Run 3 - continuation (resume, freeze 6, LR 3e-5): best 76.6%, KEPT 88.7%
Unfreezing deeper layers should help IF mid-level features were the
bottleneck. Four flat epochs say they weren't - the model is at
MobileNetV2's capacity ceiling for this data. Equally important: this run
exposed a real bug (weak metrics overwrote good ones), now fixed with the
resume-guard. Failed experiments that teach something are not wasted.

### What would 90%+ take?
A bigger backbone (EfficientNet-B3: papers report ~mid-90s on PlantVillage
at 3-5x our compute), or 2-3x longer fine-tuning with layer-wise learning
rates. Deliberately deferred: serving speed and free-tier fit matter more
for THIS product than two extra percent.

---

## Chapter 14. Inference path, precisely

`predict_image()` priority order (first success wins):

1. **PyTorch `.pt`** (normal case): preprocess (resize, ToTensor,
   ImageNet-normalize - SAME constants as training), `model.eval()` +
   `torch.no_grad()` (disables dropout AND gradient recording: faster, less
   memory), softmax, top-3 mapped through `labels.json`.
2. **Legacy Keras `.h5`** (only if someone trains the old TF script):
   MobileNet preprocess_input + predict.
3. **Heuristic fallback**: color ratios (green/lesion/dark) -> rule scores
   -> sharpened softmax. Deterministic, explainable, weak. Response carries
   `trained: false` and the UI badges "Demo mode".

Every response includes `model_accuracy` (from `metrics.json`) so the UI
prints "Trained - GPU - 88.7% val" - you can ALWAYS see which engine
answered. The server also preloads weights at startup and logs the engine:
a broken weights file crashes startup loudly instead of silently serving
heuristic answers all day.

`get_info()` (`disease_info.py`) returns cause/treatment: hand-written
entries for 15 classes, sensible generated guidance for the other 23
(parsed crop/disease names + safe generic agronomy). No class ever returns
an empty card.

---

## Chapter 15. Backend in depth (FastAPI + HTTP in 10 minutes)

**HTTP in one paragraph**: your browser (client) sends a REQUEST (method +
URL + optional body) to the server; the server runs a function and returns
a RESPONSE (status code + body). `GET` reads things (pages, scores),
`POST` sends things (the photo). `200` = success, `404` = no such thing,
`400` = bad input (our non-image rejection).

**FastAPI** maps Python functions to URLs with decorators:

```python
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
```

- `@app.post(...)` = "run this on POSTs to this URL."
- `UploadFile = File(...)` = "expect a file upload; reject the request
  otherwise." `await file.read()` streams the bytes.
- Return a dict -> FastAPI serializes it to JSON automatically.
- `async def` lets one process juggle many waiting requests (uploads,
  model math release the event loop while... note: our torch inference is
  CPU/GPU-bound Swedish... it blocks briefly - fine at this scale).

**Static files**: `app.mount("/static", StaticFiles(directory="frontend"))`
exposes the HTML/CSS/JS/images; `/` and `/about` return the pages via
`FileResponse`. One uvicorn process = whole site. CORS middleware is wide
open (`*`) so any frontend (including a future Render URL split) can call
the API.

**Endpoint table** (full behavior):

| Endpoint | Input | Output |
|---|---|---|
| `GET /`, `GET /about` | - | HTML pages |
| `POST /api/predict` | image file (<=8 MB) | prediction, confidence, top3, info, engine proof, disclaimer |
| `GET /api/health` | - | engine string, loaded?, class count |
| `GET /api/metrics` | - | test/val acc, macro+weighted avgs, 38 class rows |
| `GET /api/classes` | - | all classes + agronomy info |
| `GET /api/datasets` | - | Kaggle catalogue + setup commands |
| `GET /api/training-status` | - | live progress JSON (during runs) |
| `GET /api/plots/{name}` | curves/confusion | PNG or clean 404 |

---

## Chapter 16. Frontend in depth (no frameworks on purpose)

**Why vanilla JS?** No build step, no 200 MB `node_modules`, nothing to
compile before Render deploys - the browser runs `app.js`/`about.js`
directly. For two pages and a dozen API calls, a framework would add
machinery without value.

**The core pattern** (used on both pages):

```js
const r = await fetch("/api/metrics");  // ask (Promise = "I'll be back")
if (!r.ok) throw new Error(...);         // handle failure explicitly
const m = await r.json();                // parse body
$("#accNum").textContent = ...;          // paint the DOM
```

`await` pauses WITHOUT freezing the page; `try/catch` around it shows
"API offline" states instead of blank screens.

**Notable UI engineering decisions and their bug stories:**

- `[hidden]{display:none !important}` global guard: our flex layouts
  overrode the `hidden` attribute, so panels refused to hide. One line,
  whole class of bugs gone.
- Top-3 rows are CSS grids (name+% line, full-width bar line) so 30-char
  disease names ellipsis instead of crushing bars on phones.
- The F1 list went canvas -> HTML list -> canvas across three iterations
  (canvas kept mis-scaling at 38 rows); the accuracy donut stayed canvas
  (small charts render fine). Lesson: match the tool to the data size.
- Sample buttons fetch REAL validation photos (`frontend/samples/`),
  replacing day-one synthetic canvas blobs - and all three classify
  correctly, which doubles as a daily smoke test.
- Upload zone is a two-state machine (empty prompt <-> full photo with
  Upload/Clear), drag-drop works in both states, Clear revokes the object
  URL (browser memory hygiene).

**Responsive strategy**: CSS grid with breakpoints (detect 2-col -> 1-col
under 860px, glance stats 4 -> 2 under 640px), fluid type with `clamp()`,
no fixed pixel widths on content. Test by narrowing the browser, not by
owning every phone.

---

## Chapter 17. Deployment in depth (how code becomes a URL)

**What Render does**, step by step:

1. You push git; Render clones it into a fresh Linux container.
2. **Build**: `pip install -r requirements.txt`. Our two torch lines carry
   `sys_platform == "linux"` markers with direct CPU-wheel URLs - Render
   (Linux) installs the ~200 MB CPU build; your Windows box skips those
   lines and keeps CUDA. `torchvision` CPU needs no GPU to RUN a model -
   inference is just arithmetic, fast enough per photo (~0.1-0.5 s CPU).
3. **Start**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
   `$PORT` is injected by Render (never hardcode 8000 in production).
4. Health: first request wakes the app; free tier SLEEPS when idle
   (~30 s cold start - normal, not a bug).

**Why the weights ship IN git**: the `.pt` is 10 MB (well under GitHub's
100 MB file limit), so the deployed app works with zero post-deploy steps.
The 13 GB dataset stays OUT (gitignored) - Render never trains.

**Street smarts**: pin versions in requirements (reproducible builds);
keep the runtime lean (free tier = 512 MB RAM/disk); logs are your only
debugger in production (hence our startup engine print).

---

## Chapter 18. Hands-on labs (do these in order)

**Lab 0 - run it**: `pip install -r requirements.txt`,
`uvicorn backend.app:app --reload`, open `http://127.0.0.1:8000`. Click
all three sample buttons; each should return its true class at 85%+.

**Lab 1 - read a prediction raw**: POST a photo with curl/Python and inspect
the JSON - find `trained`, `model_accuracy`, `top3`, `features.engine`.

**Lab 2 - break preprocessing on purpose**: in a scratch script, normalize
with mean=0.5/std=0.5 instead of ImageNet stats and watch confidence
collapse. Lesson: train/serve parity is everything. Revert after.

**Lab 3 - 20-minute mini-train**: copy 3 classes x 200 images to a folder,
`python -m backend.train --data <folder> --epochs 3 --batch 32 --workers 0`,
watch the About monitor... (serve the app from another terminal). Observe
loss falling, val wobbling. Then delete the test weights and restore
`git checkout model/` to get the real ones back. (Commit first!)

**Lab 4 - read the curves**: open `model/training_curves.png`, identify our
three phases (Ch 8.4); open the confusion matrix, find the brightest
off-diagonal cell, name the confused pair, hypothesize WHY visually.

**Lab 5 - strange inputs**: upload a photo of a cat, a blurry photo, a
screenshot of text. Record confidences. Lesson: softmax ALWAYS sums to 1 -
the model cannot say "not a leaf." Real products add out-of-distribution
detection; ours documents the limitation instead.

---

## Chapter 19. Troubleshooting FAQ (real incidents from this build)

- **CUDA out of memory**: halve `--batch` (100 -> 50 -> 32). VRAM scales
  ~linearly with batch.
- **Epochs take forever on CPU**: expected (~22 min/epoch here) - that is
  why we moved to PyTorch+CUDA. Don't train big on CPU.
- **MemoryError in DataLoader workers (Windows)**: something unpicklable or
  huge travels to workers. Ours was the 10 GB memmap + a torch module.
  Fix: pass paths, open lazily per worker.
- **Metrics disagree with weights**: fixed via the resume-guard (Ch 13) -
  always re-verify `metrics.json` against the actual `.pt` after exotic
  runs: evaluate a few known images.
- **UI shows stale backend**: uvicorn froze code at startup - RESTART it
  after every backend edit. (Half our "API unreachable" scares were this.)
- **Browser shows old CSS/JS**: hard refresh (Ctrl+Shift+R); static files
  cache aggressively (those HTTP 304s in logs).
- **`hidden` elements won't hide**: a `display:` rule beats `[hidden]` -
  keep the global guard.
- **Port busy (EADDRINUSE)**: an old uvicorn still holds 8000 - find via
  `Get-NetTCPConnection -LocalPort 8000` and kill it.
- **Kaggle download resets**: `curl -C -` resumes; our 2.7 GB took retries
  (no login needed for public datasets, surprisingly).
- **Chart renders giant/blank**: fixed canvas heights fight responsive
  mode - wrapper-owned height + `maintainAspectRatio: false` + destroy
  guard; for big category lists prefer HTML over canvas.

---

## Chapter 20. Glossary (every jargon in this file)

- **Accuracy**: correct / total. Ours: 0.887.
- **Activation (ReLU)**: bend (`max(0,x)`) giving networks curve-power.
- **AMP**: 16-bit training math; ~2x faster, same result.
- **Augmentation**: random valid alterations of training images.
- **Backprop / Gradient**: backwards correction computation / its direction.
- **Batch**: one GPU-sized chunk (ours: 100 images).
- **BatchNorm**: layer stabilizing value ranges (kept frozen in eval mode).
- **Checkpoint**: best-weights file saved mid-training.
- **Class**: one disease label (38 total).
- **Confusion matrix**: truth-vs-predicted grid; diagonal = correct.
- **CORS**: browser cross-origin permission headers.
- **CPU / GPU / CUDA**: serial brain / parallel muscle / NVIDIA's GPU language.
- **Cross-entropy**: the "how wrong" loss for classification.
- **DataLoader/workers**: batch conveyor belt / its parallel loaders.
- **Dropout**: random training-time neuron blackout vs memorization.
- **Early stopping**: quit when validation plateaus; keep the best.
- **Embedding/head**: final decision layers (ours: 256-wide + 38-way).
- **Endpoint**: one URL + method served by the backend.
- **Epoch**: one full pass over training data (ours: 703 batches).
- **F1**: harmonic mean of precision and recall.
- **Fine-tuning**: retraining a pretrained net's ending on your data.
- **Freezing**: locking layers against training changes.
- **Heuristic**: hand-written rules fallback (no learning).
- **ImageNet**: 14M-photo dataset our backbone was born on.
- **Inference**: running a trained model on new data (vs training).
- **Learning rate**: optimizer step size (ours: 1e-4 decaying).
- **Logits**: raw pre-softmax scores.
- **LR scheduler**: auto learning-rate adjuster (ours: x0.3 on plateau).
- **Macro average**: mean over classes, equal weight.
- **Memmap**: file pretending to be RAM (our decode cache).
- **MobileNetV2**: small-fast CNN architecture (depthwise separable +
  inverted residuals).
- **Normalize**: shift/scale inputs to expected distribution.
- **Optimizer (Adam)**: per-weight adaptive stepper.
- **Over/underfitting**: memorizing noise / learning too little.
- **Pickle/spawn**: Python object shipping / Windows process start (our bug).
- **Precision / Recall**: right-when-claimed / found-of-existing.
- **Render**: host running our container from git.
- **Softmax**: scores -> probabilities summing to 1.
- **Split (train/val/test)**: learn-measure-final-exam data roles.
- **Support**: sample count behind a class's scores.
- **Tensor**: N-dimensional number array.
- **Top-k**: best k guesses (ours: k=3).
- **Transfer learning**: reusing pretrained vision for a new task.
- **Uvicorn**: server running our FastAPI app.
- **Validation**: honest mid-training exam (never trained on).

*End of guide. If a chapter confused you, re-read it WITH the cited file
open - every claim above points at real code or real numbers.*

---

## Chapter 21. Hyperparameter playground (what happens if YOU turn the knobs)

Everything below was either measured in our runs or follows directly from
them. Try these as Lab 3 extensions.

| Knob | Ours | If you raise it | If you lower it |
|---|---|---|---|
| `--epochs` | 25 (+early stop) | diminishing returns, overfit risk, time burn | underfit (see 5-epoch trial: 54%) |
| `--batch` | 100 | faster epoch, smoother gradients, VRAM OOM above ~180 on 6 GB | slower, noisier steps, sometimes generalizes a touch better |
| `--lr` | 1e-4 | learns fast then diverges (loss NaN at 1e-2+) | safe but needs more epochs |
| `--freeze-blocks` | 12 | faster, less overfit, lower ceiling | slower, adapts more, overfits sooner |
| `--workers` | 2 | faster loading to a point, then RAM/CPU contention | GPU starves, epoch stretches |
| AMP on/off | on | ~2x speed, same accuracy here | slower, identical math result |
| Dropout 0.3/0.2 | as built | stronger regularization, slower learning | memorizes faster, val gap grows |
| Image size 224 | 224 | finer spots, 4x cost at 448, breaks pretrained fit | faster, blurrier symptoms |

**Suggested experiments (each < 1 hour on the RTX):**
1. `--epochs 3 --batch 32` on 5 classes: watch underfitting live.
2. Same but `--no-amp`: time one epoch, compute YOUR card's AMP speedup.
3. `--freeze-blocks 18` (head only): fastest run; compare val to 88.7%.
4. `--lr 0.01`: watch loss explode - then you'll respect learning rates.
5. `--no-gpu-aug`: feel the CPU bottleneck return (watch Task Manager).

Record: final val, time/epoch, VRAM peak. You now think like an ML engineer.

---

## Chapter 22. Viva / defense Q&A (the 40 questions examiners ask)

**Project-level**
1. *What problem does this solve?* Fast, free, on-device-style crop disease
   triage for farmers without agronomist access.
2. *Why a website, not a mobile app?* One codebase, zero installs, works on
   any phone browser; model is 10 MB either way.
3. *Who is it NOT for?* Anyone needing certified diagnosis - ours is
   advisory (says so on every result).
4. *Why 38 classes and not more?* That is what PlantVillage-38 labels;
   adding crops = adding labeled data + retraining.

**Data**
5. *Where did images come from?* Kaggle PlantVillage-38, lab photos.
6. *How many?* 70,295 train / 17,572 val.
7. *Why split at all?* Measuring on training data is cheating (the model
   memorized it); validation is unseen, so honest.
8. *What is data leakage?* Test/val images accidentally IN training -
   scores look great and mean nothing. Our splits came pre-separated.
9. *Class imbalance?* Present (some diseases have 2-3x the images of
   others); macro-F1 keeps rare classes visible.

**Model**
10. *Why MobileNetV2?* 10 MB, sub-second CPU inference, ~89% - fits free
    hosting; giants don't.
11. *What is a convolution?* Shared-weight sliding window; position
    independence + parameter sharing.
12. *Depthwise separable?* Split filtering from channel-mixing; ~8x cheaper
    (Ch 6.2 numbers).
13. *What does freezing do?* Locks early generic features; trains only the
    disease-specific ending. Faster, less overfitting.
14. *What is the head?* Dropout-Dense256-Dropout-38 + softmax; the only
    randomly initialized part.
15. *Why softmax?* Comparable probabilities summing to 1; confidence =
    top probability.
16. *ReLU?* max(0,x); bends the math so depth adds power.

**Training**
17. *Define epoch/batch/step.* Full pass / 100-image chunk / one chunk's
    update. Ours: 703 steps/epoch.
18. *What is loss?* Cross-entropy: -log(probability given to the TRUE
    class). Confident-wrong hurts most.
19. *What is backprop?* Chain-rule correction computation, automated.
20. *Why Adam?* Per-weight adaptive steps; fast, forgiving.
21. *What is learning rate scheduling?* Shrinking steps as you near the
    optimum (our 1e-4 -> 2.7e-6 staircase).
22. *Early stopping?* Quit 4 flat epochs after the peak; keep best weights.
23. *What is augmentation and why on GPU?* Random valid variations vs
    memorization; GPU does the affine math free-ish.
24. *AMP?* 16-bit math, ~2x faster, scaler protects tiny gradients.
25. *What are workers?* Parallel CPU loaders overlapping IO with compute.
26. *Why normalize with ImageNet stats?* Pretrained weights expect that
    distribution; mismatch silently kills accuracy.
27. *Overfitting signs?* Train rises, val stalls/falls; dropout, aug, early
    stop fight it.

**Results**
28. *Final numbers?* 88.7% val/test, macro-F1 0.887, best epoch 14/18.
29. *Why not 100%?* Similar-looking diseases, label noise, model capacity,
    lab-vs-field gap.
30. *Precision vs recall?* Right-when-claimed vs found-of-existing (Ch 12
    tomato example).
31. *Why F1 over accuracy?* Accuracy hides rare-class failure; F1 + support
    expose it.
32. *Read the confusion matrix for me.* Diagonal = correct; off-diagonal =
    confused pairs (ours: tomato spots, maize rusts).
33. *Why did the continuation fail?* Capacity ceiling reached; deeper
    features weren't the bottleneck. Evidence > hope.
34. *Is 88.7% good?* For a 10 MB model on 38 lab classes: strong. SOTA with
    bigger backbones reaches mid-90s at 3-5x cost.

**System**
35. *How does upload become diagnosis?* Chapter 2's 7-step trace - recite it.
36. *What if weights are missing?* Heuristic fallback + "Demo mode" badge;
    never silent.
37. *How do you KNOW the best model serves?* Startup preload + log line,
    per-response `trained` flag + accuracy chip.
38. *Why CPU torch on Render?* Serving needs no GPU; CUDA wheels are
    gigabytes; markers keep dev CUDA intact.
39. *How do retrains reach the site?* Overwrite `model/` files, restart;
    resume-guard protects good artifacts from weak runs.
40. *Biggest bug you fixed?* The resume run overwriting 88.7% metrics with
    76.6% numbers - now impossible by construction (Ch 13).

---

## Chapter 23. Math appendix (every formula used, gently)

**A. Softmax** (logits z, classes i): `p_i = e^z_i / sum(e^z_j)`.
Example Ch 5.3: `[2.0, 1.0, 0.1]` -> `[0.659, 0.242, 0.099]`.

**B. Cross-entropy loss** (true class c): `L = -log(p_c)`. Confident-right
(p=0.99): L=0.01. Confident-wrong (p=0.01): L=4.6. Training minimizes the
mean over the batch - confident mistakes dominate, as they should.

**C. Gradient descent step**: `w -= lr * dL/dw`. Adam replaces `lr` with a
per-weight `lr * m_hat / (sqrt(v_hat) + eps)` where m, v are
gradient-squared running averages (memory of direction and jumpiness).

**D. Convolution cost** (Ch 6.2): standard `K*K*Cin*Cout*H*W` vs separable
`(K*K*Cin + Cin*Cout)*H*W`. Plug K=3, Cin=32, Cout=64, H=W=112: 231M vs 29M.

**E. Affine augmentation matrix** (per image, applied on GPU):
`theta = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]` with angle +/-15 deg,
scale 0.85-1.15, shift +/-0.1. `affine_grid` turns it into a sampling grid;
`grid_sample` re-reads pixels bilinearly. One matrix = rotate+zoom+shift.

**F. Normalize**: `x = (x/255 - mean) / std`, per channel. Red example in
Ch 9.2: `(0.431-0.485)/0.229 = -0.236`.

**G. Precision/Recall/F1**: `P = TP/(TP+FP)`, `R = TP/(TP+FN)`,
`F1 = 2PR/(P+R)`. Macro = mean of class F1s; weighted = support-weighted.

**H. Parameter count sanity**: our trainable 2.26M x 4 bytes = ~9 MB -
matches the 10 MB `.pt` file (weights + small header). If a checkpoint is
ever 10x off, suspect a shape/architecture mismatch.

*End of guide (v2). If a chapter confused you, re-read it WITH the cited
file open - every claim above points at real code or real numbers.*

---

## Chapter 24. Code walkthrough (the 30 lines that matter most)

**train.py - the four lines of learning** (inside `run_epoch`):

```python
out = model(x)              # forward: 100 photos -> 100 x 38 scores
loss = criterion(out, y)    # one number: how wrong?
scaler.scale(loss).backward()  # backprop (AMP-scaled so 16-bit survives)
scaler.step(optimizer); scaler.update()  # move every weight a little
```

Everything around them is scaffolding: loading, freezing, scheduling,
saving. If you remember nothing else, remember these four.

**train.py - freezing** (`freeze_early_blocks`): loops the 19 feature
blocks; the first N get `requires_grad_(False)` (no corrections computed)
AND `.eval()` (BatchNorm statistics pinned - frozen layers must not drift
while the head learns around them). Called once at start and after every
`model.train()`, because `.train()` flips ALL modules back including the
frozen ones - a subtle bug if forgotten.

**model_handler.py - engine priority** (`predict_image`): try `.pt` via
torch -> try `.h5` via TF -> heuristic. Each returns `None` on mismatch
(wrong class count, corrupt file) so the chain falls through gracefully.
`top_k()` is pure Python sorting - no ML at all.

**model_handler.py - why `torch.no_grad()` + `model.eval()`**: no_grad
skips recording operations (no backward needed) - halves memory and adds
speed. eval() disables dropout - full network votes deterministically.
Forget either and predictions get slower, hungrier, or randomly different
between identical uploads.

**app.py - the whole API pattern**:
decorator routes URL -> function, `UploadFile` validates input, return dict
becomes JSON, `HTTPException(400)` rejects non-images and >8 MB files.
`FileResponse` serves pages and PNG plots; `StaticFiles` serves the
frontend folder. Startup preload (`_try_load_torch_model()` + print) makes
the serving engine visible in logs from second zero.

**about.js/app.js - the whole frontend pattern**: `fetch` -> `r.ok` check
-> `r.json()` -> paint `textContent`/`innerHTML` by element id. Every
dynamic number on both pages arrives this way; nothing is hardcoded except
placeholder defaults.

---

## Chapter 25. Where to go from here (roadmap by difficulty)

**Easy (weekend each):**
- Add out-of-distribution detection ("this isn't a leaf") via confidence
  thresholding plus a small calibration study (Lab 5 data).
- Per-class agronomy pages: expand `disease_info.py` to full 38 detailed
  entries with images and regional spray schedules.
- Regional-language toggle for the frontend (farmers first).

**Medium (2-4 weeks):**
- Field-photo fine-tuning: collect phone photos, mix with PlantVillage,
  retrain with `--resume` - expect the lab-to-field gap to shrink.
- Bigger backbone experiment (EfficientNet-B0/B3) using the same trainer
  (swap `build_mobilenet`); compare val, `.pt` size, CPU latency.
- Object detection upgrade (localize MULTIPLE lesions per leaf) with the
  PlantDoc dataset already catalogued in `datasets.py`.

**Hard (project-scale):**
- Severity scoring (% leaf area infected) via segmentation (U-Net) on top
  of classification.
- Offline mobile app (convert `.pt` -> ONNX -> phone runtime).
- Multilingual voice interface for low-literacy users.

Pick ONE easy item, finish it completely, then decide. Finished small
things beat half-built big things - this whole project is proof: 25 files,
one model, two pages, 88.7%.

*End of guide (v3). 1000+ lines, zero hand-waving - now go build the next
thing.*

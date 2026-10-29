# EM Brain Viewer: merge and split cells (prototype)

A small web app for up to 3 people. It shows a piece of the CREMI Sample A brain-tissue
picture in Neuroglancer, with the cells colored on top and 3D shapes for selected cells.
You can **merge** two cells, **split** one cell in two, **undo**, and your fixes are
saved so they are still there after a restart.

The plan and the reasons behind each choice are in `PLAN.md`.

> **Important: what has and has not been tested.**
> The graph, merge, split, undo, saving, settings and download code are covered by
> automated tests (`pytest`, 49 tests). The code that talks to the *real* Neuroglancer
> and the *real* HDF5 file (`src/viewer.py`, the HDF5 part of `src/data.py`) was written
> without being able to install those packages, and was only tested against a stand-in.
> Run through "First things to check" below on your first start.

---

## Using the viewer

Open your link, then:

1. In the segmentation layer, click a cell to select it (the 3D panel shows its shape).
2. Use the keys below (hover the mouse over the picture first when a key needs a position).

| Key | What it does |
|---|---|
| `Shift+M` | **Merge** the two selected cells into one |
| `Shift+A` | Add a **seed A** where the mouse is (press several times for several seeds) |
| `Shift+B` | Add a **seed B** where the mouse is |
| `Shift+X` | **Split**: cut the thinnest connections that separate the A seeds from the B seeds |
| `Shift+C` | Clear your seeds |
| `Shift+U` | **Undo** the last edit (by anyone) |

How to split: put one or more A seeds on one part of the cell and one or more B seeds on the
part you want separated, then press `Shift+X`. The cut follows the thinnest "neck" between them.

Messages appear at the top of the viewer. Everyone sees everyone's edits (they appear after
the next redraw). In `per_user` mode each person has their own camera and selection.

---

## Run on your computer

You need Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                   # optional: 49 tests, takes a second

# quick try with a tiny crop (the first start downloads about 180 MB into ./data)
CROP_SIZE=16,128,128 python src/app.py
```

The program prints one link per person, like `http://localhost:8080/v/<token>/`. Open one in
your browser. Tokens are random unless you set `VIEWER_TOKENS` (see Settings), so the links
change each start unless you set it.

Normal size crop: `python src/app.py` (default 64 x 512 x 512 voxels).

## Run with Docker

```bash
docker build -t em-viewer .
docker run --rm -p 8080:8080 -v "$PWD/data:/data" \
  -e VIEWER_TOKENS=alice-secret-123,bob-secret-456,carol-secret-789 em-viewer
```

The data file and the saved edits live in `./data` on your computer (inside the container: `/data`),
so they stay when the container stops.

## Deploy to Fly.io

1. Install the Fly CLI and run `fly auth login`.
2. In `fly.toml`, change `app = "..."` to a name nobody else has, for example `em-viewer-yourname`.
   Pick a `primary_region` near you (`fly platform regions`).
3. Create the app and its saved disk (one machine, one disk):
   ```bash
   fly launch --no-deploy --copy-config
   fly volumes create em_data --region bos --size 1
   ```
4. Set your secrets (the secret link words and the public address):
   ```bash
   fly secrets set \
     VIEWER_TOKENS="$(python3 -c 'import secrets;print(",".join(secrets.token_urlsafe(12) for _ in range(3)))')" \
     PUBLIC_URL="https://<your-app-name>.fly.dev"
   ```
5. Deploy exactly one machine (the edits live in that machine's memory and disk):
   ```bash
   fly deploy --ha=false
   fly status          # should show 1 machine
   fly logs            # the links are printed here when it is ready
   ```
   The first start downloads the 180 MB data file, so give it a couple of minutes.
6. Restart test: `fly machine restart <machine-id>`, wait, open the link: your edits should still be there.

**Never run more than one machine.** Two machines would each have their own memory and disk and would disagree.

**Safety:** there is no login. Anyone who has a link can edit. The long random words in the links
are the only protection, so share links only with people you trust, and keep them out of public places.
Read the links from `fly logs` (private to you).

### Render (backup)

Use the same Dockerfile. You need a paid plan with a persistent disk mounted at `/data`
(the free plan sleeps and loses files). Render gives the port in `$PORT`; set `PUBLIC_URL`
to your `https://<name>.onrender.com` address and `VIEWER_TOKENS` as above.

---

## Settings (environment variables)

| Name | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` (Docker: `/data`) | Where the HDF5 file and `edits.sqlite` live |
| `PORT` | `8080` | Port to listen on |
| `PUBLIC_URL` | `http://localhost:$PORT` | The address people open, e.g. `https://myapp.fly.dev` |
| `VIEWER_MODE` | `per_user` | `per_user`: one viewer per token. `shared`: one viewer for everyone |
| `VIEWER_TOKENS` | random | Comma list of secret link words (letters, digits, `-`, `_`; 12+ characters is wise) |
| `CROP_START` | `30,300,300` | Where the crop starts (z,y,x) |
| `CROP_SIZE` | `64,512,512` | Crop size (z,y,x). Smaller is faster and uses less memory |
| `BLOCK_SIZE` | `8,64,64` | Brick size (z,y,x) used to cut cells into small pieces |
| `DATASET_URL` | CREMI Sample A link | Where to download from if the file is missing |
| `HDF_NAME` | `sampleA__20160501.hdf` | File name inside `DATA_DIR` |

If you change `CROP_START`, `CROP_SIZE` or `BLOCK_SIZE`, the saved edits no longer fit. The app
notices, moves the old log aside as `edits.sqlite.mismatch-<time>` and starts with no edits.

---

## First things to check (not possible to test where this was written)

Do these in this order on your first run, and tell me what you see:

1. **App starts and prints links.** If `pip install` fails for `neuroglancer`, add `build-essential` in the Dockerfile (comment inside).
2. **The page loads** at the printed link, and also at `https://<app>.fly.dev/v/<token>/` on Fly. If the logs say "Unexpected viewer URL shape", Neuroglancer is using its hosted web page; tell me.
3. **Shape of the cells.** They should look natural, not squashed. (Voxel size is 40 x 4 x 4 nm, read from the file, and the viewer uses x,y = 4 nm and z = 40 nm.)
4. **Keys.** Press `h` in Neuroglancer to see its key list and make sure none of the `Shift+` keys above clash. Press `Shift+A` over a cell: a message should say "Seed A on object N". If it says "No cell under the mouse" on a cell, the mouse position is not in voxel units; tell me.
5. **Merge / split / undo** change the colors in 2D and the 3D shape updates (select the cell so its shape is shown). If colors change but the 3D shape does not, tell me.
6. **Labels.** In the logs look for "Crop has N neurons; X% of voxels are 'no neuron'". CREMI's "no neuron" number is assumed to be the largest 64-bit number. If X is 0% or 100%, tell me.
7. **Memory.** `fly status` / `fly logs` should not show the machine running out of memory. If it does, use a smaller `CROP_SIZE` or `fly scale memory 2048`.

---

## Files

```
src/app.py       start-up, prints the links
src/config.py    settings from environment variables
src/data.py      download, crop, clean the labels
src/graph.py     bricks, glue, objects, cheat sheet, merge pair, thinnest cut
src/store.py     SQLite edit log (undo, restart replay, data fingerprint)
src/service.py   merge / split / undo (no Neuroglancer inside)
src/viewer.py    Neuroglancer layers, key actions, one viewer per user
tests/           automated tests (a made-up tiny brain, plus a stand-in for Neuroglancer)
```

Names in the edit log are `user1`, `user2`, ... (the order of the tokens), never the secret tokens.

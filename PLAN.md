
---

# BUILD NOTES (what changed while building)

- **Settled in the code:** user names in the edit log are `user1`, `user2`... (never the secret tokens). Edit ids, undo and replay work as in section 9.
- **Selection follows edits.** After an edit, the person who edited sees the new object(s) selected. Everyone else keeps their selection: an object's number is its smallest brick number, so each old selection is moved to whichever object now holds that brick.
- **Memory fix.** Cleaning the labels on a full 64x512x512 crop first used about 680 MB; it now works one slice at a time (about 350 MB peak for the whole start-up on a made-up full-size crop).
- **Fingerprint** also contains a hash of the brick picture, so any change in how bricks are cut is noticed.
- **Weakest glue rule:** glue added by hand between bricks that do not really touch has the lowest strength, so a later split will undo it first (tested).
- **Not verifiable where this was written (no access to PyPI):** real Neuroglancer behavior (key bindings, mouse coordinates in voxel units, mesh refresh after `invalidate()`, fixed tokens, link shape behind Fly's https) and the real HDF5 file (background label value, resolution attribute). The checklist is in README.md, "First things to check".

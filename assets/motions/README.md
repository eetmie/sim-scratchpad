# Small excavator motion demo

`excavator_demo.json` contains 110 poses over about 22 seconds, at the original
approximately 5 Hz exposure cadence. Run `launch_3dgs.py --mode replay` to loop it with
live rendered depth and heightmaps. `--replay assets/motions/excavator_demo.json`
selects this same camera/map replay explicitly.

It is a compact copy of the validated episode-002 cabin-pose timeline, frames
12 through 666 at stride 6. Only relative timestamps and five link transforms
are retained. USD matrices use row vectors and are rounded to six decimals;
the file is about 99 KiB. It contains no depth, images, recordings, or local paths.
The original replay's SHA-256 is recorded for provenance.

This is paused, kinematic motion. It does not drive hydraulic physics or deform
the scanned sand. Each wrap returns to the first pose and clears the accumulated
map. The source recording is not needed to play it.

# SMPL UV and sample texture attribution

The original OBJ/MTL samples and the two unmodified albedo PNGs were published by
Meshcapade in [SMPL_texture_samples](https://github.com/Meshcapade/SMPL_texture_samples),
at fixed commit `c87d871c7498fdc98f3cabb7341b5c85c38debaa`.

They are licensed under the [SMPL-Body Creative Commons Attribution 4.0
International license](https://smpl.is.tue.mpg.de/bodylicense.html), whose terms
link to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Licensor: Max Planck Society e.V. Credit: Max Planck Institute for Intelligent
Systems, and Meshcapade for the published sample textures.

For media projects: “SMPL-Body was used for character animation courtesy of the
Max Planck Institute for Intelligent Systems.” For publications, cite the most
recent SMPL model paper specified at <https://smpl.is.tue.mpg.de/>.

## Files and changes

- `meshcapade_source/SMPL_male.obj`, `SMPL_female.obj`, and both MTL files preserve
  the original bytes. These are subdivision samples, not the runtime geometry.
- `textures/f_01_alb.002.png` and `m_01_alb.002.png` preserve the original 4096×4096
  RGB images without resizing. Only albedo images belong in the runtime pool;
  normal maps and clothing preview images were not installed.
- `smpl_uv.npz` is derived from the male sample OBJ. Its subdivision topology was
  contracted to original SMPL parent faces, then UV corner indices were reordered
  to match all 13,776 triangles of the existing 6,890-vertex neutral model.
  UV coordinate values were preserved; male/female mapped corner coordinates are
  identical. This is a derived UV archive, not a claim that the OBJ itself has
  6,890 vertices or that an original downloadable UV archive was obtained.
- The existing `SMPL_NEUTRAL.pkl` and its original model license were not changed.
  The sample body's CC BY license does not relicense that parametric model.

Conversion: `docs/resource_completion_20261001/install_meshcapade_smpl_uv.py`.
Exact source sizes, SHA-256, Git blob SHA-1 and topology checks:
`docs/resource_completion_20261001/smpl_uv_geometry_verification.json` and
`smpl_texture_installation.json`.

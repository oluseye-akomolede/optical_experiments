# time-domain

## ADDED Requirements

### Requirement: The field movie SHALL draw the uploaded object over the field

The field movie of a finished time-domain job SHALL draw the uploaded object's
outline -- every part of the upload, its edges and a faint shade of its
surfaces -- in the same 3-D view as the field, on every frame it shows, while
playing and while scrubbing. The outline's parts SHALL be placed by the job's
own `units` and `scale` alone (the upload's own coordinates multiplied by
`unit_scale(units) x scale`), with no further translation and no further
scaling: the movie's frames are stored in exactly that frame, whose origin is
the parts' combined bounding-box minimum. A job whose outline cannot be built
or fetched SHALL still play its movie and SHALL say in the caption that the
object outline could not be drawn and why -- never a movie with no object and
no explanation, and never a movie that stops.

Why: the owner asked for the uploaded object to be visible during the movies
(2026-10-02), and a frame drawn without it is a field in an empty box -- the
box's wireframe, not the object.

#### Scenario: an outline on every frame

- **WHEN** a finished td job with a movie is opened and its frames are stepped
  through and played
- **THEN** the uploaded object is drawn over the field on every frame shown,
  at the position the job's own units and scale put it

#### Scenario: an outline that cannot be drawn is named

- **WHEN** the job's outline cannot be built or fetched
- **THEN** the movie still plays its frames and the caption says the object
  outline could not be drawn, naming the reason

### Requirement: A job's object outline SHALL come from its stored upload

The uploaded object's outline SHALL be derivable from a job's own stored upload
and the job's own `units` and `scale` at the moment it is asked for, rather than
from anything the solver writes: every part of the upload SHALL be drawn, parts
SHALL be scaled by the same function and factor the solver used, and no part
SHALL be dropped or overwritten by another part's name. Jobs solved before this
existed SHALL therefore show their outline with no re-run, and the solver SHALL
need no change.

Why: the cavity view's surface file is written only by the cavity solver, in
the user's units and centred on its own origin, so it is not the object's place
in the movie's frame; deriving the outline on request is what lets the owner's
already-solved film job show its object.

#### Scenario: an already-solved job shows its object

- **WHEN** a job solved before this change is opened
- **THEN** its movie draws that job's upload, with no re-run

#### Scenario: a multi-part upload keeps every part

- **WHEN** the upload holds more than one part
- **THEN** every part is in the outline, at its own place

### Requirement: The movie page SHALL state where it drew the outline

The movie page SHALL expose, for the frame it is showing, the axis-aligned box
of the outline it actually added to its scene, in metres, in a form a check can
read back -- and SHALL state that no outline was drawn, rather than nothing,
when the outline could not be built.

Why: an outline drawn near the object instead of on it is the defect the owner
reported, and a check that only counts pixels cannot tell the two apart.

#### Scenario: an outline that is drawn

- **WHEN** the page draws the outline
- **THEN** the box it states is the outline's own, in metres

#### Scenario: an outline that is not drawn

- **WHEN** the outline cannot be fetched
- **THEN** the page states that none was drawn, and the caption says why

### Requirement: The render check SHALL verify the outline on every sampled frame

The render check SHALL read the box the page states it drew and require it to
be the box the manifest's own sample positions span, within one grid cell, for
a mirrored real job -- so a unit, a scale or a translation error in the
outline's frame is a failure and not a matter of opinion. It SHALL further
draw every frame it samples on the same page twice: once with the outline and
once with the outline route answering 404, the two runs differing in nothing
else, and require each sampled frame to gain a substantial number of pixels
from the outline -- including a frame reached by playing the movie, not only
by scrubbing -- while the run without the outline still draws its field. The
share of the frame's own lit pixels the outline must add SHALL be a kind floor
well below what a drawn object measures, not a pass mark set at the
measurement.

Why: a box compared against a centred object would pass for an outline drawn in
the grid's frame or shifted to the box's centre, so the comparison is only worth
making on an object that is off-centre -- the check's own fixture is therefore
two parts, the second at negative x, and the mirrored real job (the owner's film
job) is the same case on production's own artifacts. The pixel pair is what
shows the object is drawn rather than merely described and placed: the stated
box is read from the group the page built, which it builds whether or not it
adds it to the scene.

#### Scenario: the outline sits on the field's own frame

- **WHEN** the check runs against a mirror whose object is not centred in its
  domain
- **THEN** the box the page states for the outline and the box the manifest's
  positions span agree within one grid cell, and a translated or rescaled
  outline FAILs naming the axis

#### Scenario: the outline is drawn, not merely present

- **WHEN** the same frame is drawn with and without the outline
- **THEN** the frame with it has substantially more drawn pixels, on every
  sampled frame including one reached by playing, and the run without it is
  still a drawn field rather than a blank frame

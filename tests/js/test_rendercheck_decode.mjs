// Check the check's own instrument: the node-side PNG decode.
//
// The render check now captures by asking the page for a PNG data URL
// (`canvas.toDataURL('image/png')`) and decoding it here, instead of reading
// pixels inside the page. That moved the whole pixel path onto `decodePng`, and
// a decoder that is quietly wrong makes every frame the check judges a fiction.
// Wrong in the direction of *fewer* lit pixels is exactly a predicate that
// reports green -- which is the failure this row exists to remove -- so this
// file exists to catch a decoder bug before the render check believes it.
//
// The fixture below is NOT built by this file. It is a real PNG produced by
// Chromium's own encoder (`canvas.toDataURL`), 4x2, drawn as: row 0 solid red,
// row 1 the page background with one blue pixel. A fixture constructed here
// would let the test agree with its own arithmetic; this one cannot.

import test from 'node:test';
import assert from 'node:assert/strict';
import { decodePng } from '../../dockerfiles/rendercheck/pixels.mjs';

// Chromium-written, colour type 6 (RGBA), bit depth 8, one IDAT.
const BROWSER_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAQAAAACCAYAAAB/qH1jAAAAAXNSR0IArs4c6QAAAB5JREFUGFdj' +
  '/M/A8J8BCTBy8wmBBb5+esvAzSfMAABi1wYrHhhuvQAAAABJRU5ErkJggg==';

test('decodePng reads the bytes a canvas read hands it', async () => {
  const png = Buffer.from(BROWSER_PNG_BASE64, 'base64');
  const frame = await decodePng(png);

  assert.equal(frame.width, 4, 'the fixture is 4 px wide');
  assert.equal(frame.height, 2, 'the fixture is 2 px tall');
  assert.equal(frame.data.length, 4 * 2 * 4, 'RGBA quads, one per pixel');

  // Row 0, pixel 0: red.
  assert.deepEqual([...frame.data.subarray(0, 4)], [255, 0, 0, 255]);
  // Row 1, pixel 0: the page background the predicates measure against.
  assert.deepEqual([...frame.data.subarray(16, 20)], [11, 14, 18, 255]);
  // Row 1, pixel 2: the one blue pixel.
  assert.deepEqual([...frame.data.subarray(24, 28)], [0, 0, 255, 255]);
});

test('decodePng tells the two lobes apart in the fixture it was given', async () => {
  // The predicates are hue-based (warm = r > b + 32, cool = b > r + 32), so a
  // decoder that swapped the R and B channels would leave a real frame's warm
  // and cool counts EXCHANGED and still pass a "did it draw" count. Naming both
  // members separately is what stops a swap from hiding.
  const { data } = await decodePng(Buffer.from(BROWSER_PNG_BASE64, 'base64'));
  const warm = [...data].filter((_, i) => i % 4 === 0)
    .filter((r, n) => r > data[n * 4 + 2] + 32).length;
  const cool = [...data].filter((_, i) => i % 4 === 0)
    .filter((r, n) => data[n * 4 + 2] > r + 32).length;
  assert.equal(warm, 4, 'the red row is warm');
  assert.equal(cool, 1, 'the single blue pixel is cool');
});

test('decodePng widens a type-2 (RGB) PNG to RGBA, R and B the right way round', async () => {
  // Chromium's toDataURL emits colour type 6 in this container, so the other
  // shape the format allows -- colour type 2, three channels per pixel -- is
  // never exercised by the RGBA fixture above, and a decoder that swapped R and
  // B on that path would pass every test in this file while painting every real
  // frame's warm and cool lobes backwards.
  //
  // The fixture is the SAME PIXELS as the RGBA one, transcoded to type 2 by
  // rewriting only the IHDR colour type and the IDAT payload. That it really is
  // an RGB PNG with those pixels was checked with an independent decoder (PIL
  // reports mode RGB, size 4x2, and the same three pixels) rather than with the
  // decoder under test -- a fixture this file produced and this file judged
  // would prove nothing.
  const rgb = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAQAAAACCAIAAADwyuo0AAAAFklEQVR42mP4z8AARwzcfEJABGQD' +
    'SQBZNgV95rPLggAAAABJRU5ErkJggg==', 'base64');
  assert.equal(rgb[25], 2, 'the fixture really is colour type 2');

  const frame = await decodePng(rgb);
  assert.equal(frame.width, 4);
  assert.equal(frame.height, 2);
  assert.equal(frame.data.length, 4 * 2 * 4, 'RGB is widened to RGBA quads');
  assert.deepEqual([...frame.data.subarray(0, 4)], [255, 0, 0, 255],
    'red must stay red: a swap here inverts every frame\'s lobes');
  assert.deepEqual([...frame.data.subarray(24, 28)], [0, 0, 255, 255],
    'blue must stay blue');
});

test('decodePng refuses a PNG it cannot read rather than guessing', async () => {
  const png = Buffer.from(BROWSER_PNG_BASE64, 'base64');

  await assert.rejects(() => decodePng(png.subarray(0, 4)), /not a PNG/,
    'a truncated file must not be decoded into a frame');

  // 16-bit depth, which is what a different encoder might emit: IHDR byte 8 is
  // the bit depth, and changing it to 16 must make the decode refuse, not
  // return half-length rows.
  const deep = Buffer.from(png);
  deep[24] = 16;
  await assert.rejects(() => decodePng(deep), /bit depth/);

  // Palette (colour type 3), the other shape a PNG can arrive in.
  const palette = Buffer.from(png);
  palette[25] = 3;
  await assert.rejects(() => decodePng(palette), /colour type/);
});

# Render check: does the volume plot actually draw?
#
# The unit tests cover the server's bytes and the transform's maths, and neither
# can see the one thing the owner asked for: a picture. A shader that fails to
# compile, a texture that uploads nothing, a camera that never sees the box --
# all of those leave every test green and the page blank, and a blank page is
# exactly what this row exists to prevent.
#
# So this image loads the live page in headless Chromium, drives the component
# and coordinate-system selectors, and fails unless pixels changed. It runs as a
# bounded Job in the cluster's namespace, never natively: a browser is heavy,
# and the VM has no GPU to give it.
#
# Why not Playwright's own image, which is the obvious choice and was the first
# one: `mcr.microsoft.com/playwright:v1.49.0-noble` installs all three browsers
# plus their system dependencies in a SINGLE layer, and that layer is 2.16 GB.
# Harbor rejects it on push -- 'blob upload invalid' / 'unknown error' -- while
# a 1 GB layer pushes fine. The registry's blob unit is the layer, and Docker's
# layer unit is the RUN instruction, so "install less" is not enough on its own:
# everything one RUN touches is one blob however it was installed.
#
# Hence the shape below. A slim base, Chromium only, and -- the load-bearing
# part -- the apt dependencies, the browser download, and node_modules each in
# their own RUN, so no single layer carries more than one of them. Measured
# sizes are in the comments at each step.
FROM node:20-bookworm-slim

ENV NODE_ENV=production

WORKDIR /opt/rendercheck

COPY dockerfiles/rendercheck/package.json /opt/rendercheck/package.json

# Chromium's runtime .so closure, which Playwright knows and we do not: better
# to ask the tool that ships the browser for its own dependency list than to
# hand-roll one that goes stale silently (the failure mode is not a clean error
# -- it is a browser that starts and then cannot make a WebGL context).
#
# --no-install-recommends and clearing /var/lib/apt/lists in the SAME RUN are
# both load-bearing: the .deb cache is otherwise persisted into this layer, and
# it is the apt half that would push a combined layer over the limit.
#
# This RUN is its own layer, deliberately. Keep it that way.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates fonts-liberation libasound2 libatk-bridge2.0-0 \
      libatk1.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 \
      libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 libxcomposite1 \
      libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 wget xdg-utils \
 && rm -rf /var/lib/apt/lists/*

# The browser, in its own layer again. --only-shell because the check launches
# headless, and headless Chromium uses the headless shell binary: installing the
# full browser as well would be 549 MB of executable this image never runs.
# (Measured in the prototype: chromium-1148 549 MB, chromium_headless_shell-1148
# 309 MB. Only the shell is needed; --only-shell is supported in 1.49.0.)
#
# PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD is explicitly cleared for this step. The
# earlier version of this file set it to 1 to stop the Playwright base image
# downloading browsers during `npm install`, and leaving it in force here would
# make this install silently do nothing -- a failure that shows up at runtime in
# the cluster, not at build time, which is the worst place for it.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD= \
      npx --yes playwright-core@1.49.0 install --only-shell chromium \
 && rm -rf /root/.cache/ms-playwright

# node_modules, its own layer too. `npm install` rather than `npm ci` because
# there is no lockfile in this directory to install from.
RUN npm install --omit=dev --no-audit --no-fund

COPY dockerfiles/rendercheck/check.mjs /opt/rendercheck/check.mjs
# The pixel predicates, imported by check.mjs and tested on their own in
# tests/js/test_rendercheck_predicates.mjs. Kept beside the check so the file
# the tests import is the file the check runs.
COPY dockerfiles/rendercheck/pixels.mjs /opt/rendercheck/pixels.mjs

# No root: the Job runs as a non-root user with a read-only root filesystem, so
# the browser needs a writable /tmp to put its own profile in. HOME is pointed
# at /tmp as well, because Chromium writes its profile under $HOME and the
# rendercheck user's home is on the read-only root filesystem at runtime -- the
# Job mounts a writable emptyDir only at /tmp and /out.
RUN useradd --create-home --uid 10002 rendercheck && mkdir -p /out && chown rendercheck /out
ENV HOME=/tmp
USER 10002

ENTRYPOINT ["node", "/opt/rendercheck/check.mjs"]

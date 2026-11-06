# Footy AI agent handoff

## Start here (2026-11-06)

- The Flutter APK (`D:\Projects\footyai\Footy_AI\footy_ai_app`, branch
  `sinister/experimental`) is the main client. The `backend/static` web page
  described below is kept but not developed further.
- Feature switches live in `features.yaml` (read by `feature_flags.py`, used by
  `PipelineConfig.short_clip` per job and served to the app by
  `GET /api/features`). Add a switch there rather than hard-coding a flag.
- `docs/HOW_TO_CHANGE_THINGS.md` is the plain-language map of which file to
  change; `docs/TECHNICAL_GUIDE.md` is the full developer reference (API,
  database, pipeline, app). Keep both current when you change behaviour.

## What changed

- `backend/static/index.html`, `backend/static/app.css`, and
  `backend/static/app.js` implement the responsive Footy AI web workflow:
  upload footage, confirm two player-image kit samples, then read the score
  sheet and detected scorers.
- Reference captures for that interface are committed at
  `output_videos/ui-desktop.png` and `output_videos/ui-mobile.png`.
- `DESIGN.md` is the visual-system reference and `PRODUCT.md` explains the
  intended user flow. Preserve the evidence-first team confirmation: never ask
  a user to identify a team through a hex color value.

## Localizing the screens

The current web screen is English-only. When adding another language:

1. Move every user-visible string from `backend/static/index.html` and
   `backend/static/app.js` into a locale dictionary or translation files. This
   includes button text, validation and processing messages, image alt text,
   labels, placeholders, and dynamically rendered result text.
2. Keep stable technical values unchanged: API routes, JSON field names,
   element IDs, CSS classes, `data-*` values used as program state, and status
   values returned by the backend. Translate their displayed labels only.
3. Use the document's `lang` attribute and keep the selected locale available
   during the complete upload-to-results flow. Add `dir="rtl"` and RTL CSS
   overrides for right-to-left locales.
4. Do not build translated text with partial string concatenation. Use whole
   message templates with named parameters so grammar and word order work in
   every locale. Localize dates, numbers, percent values, and durations with
   `Intl` formatting; preserve football notation where it is conventional.
5. Test each locale at the desktop and mobile widths represented by the
   committed screen captures. Allow controls and cards to expand for longer
   translations; do not truncate critical actions or team names.
6. Retain the accessibility meaning in every translation: descriptive `alt`
   text, `aria-label` values, form labels, status announcements, and visible
   error messages must remain clear and equivalent.

After localization, refresh the two reference captures if the layout changes
materially and update this handoff with the supported locale codes and how a
user selects one.

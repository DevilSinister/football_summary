---
name: Footy AI
description: A daylight analyst desk for turning match footage into a named, evidence-led score sheet.
colors:
  ink-navy: "#10233f"
  ink-soft: "#526176"
  paper: "#f7f8f4"
  canvas-white: "#ffffff"
  rule: "#d7ddd4"
  turf: "#087a52"
  turf-dark: "#04583c"
  signal-lime: "#c8f04d"
  error: "#b42318"
  intro-copy: "#cbd5e2"
  quiet-surface: "#fbfcf9"
  chip-surface: "#edf1ec"
  field-border: "#aab4aa"
  drop-border: "#9eaa9e"
  disabled: "#98a49d"
typography:
  display:
    fontFamily: "Aptos, Segoe UI, Arial, sans-serif"
    fontSize: "clamp(40px, 5vw, 72px)"
    fontWeight: 700
    lineHeight: 0.98
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Aptos, Segoe UI, Arial, sans-serif"
    fontSize: "clamp(30px, 3.3vw, 48px)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Aptos, Segoe UI, Arial, sans-serif"
    fontSize: "18px"
    fontWeight: 800
    lineHeight: 1.2
  body:
    fontFamily: "Aptos, Segoe UI, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Aptos, Segoe UI, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "0.09em"
rounded:
  subtle: "2px"
  brand: "8px"
  control: "10px"
  surface: "14px"
  pill: "999px"
  circle: "50%"
spacing:
  xs: "8px"
  sm: "12px"
  md: "18px"
  lg: "24px"
  xl: "32px"
  field: "15px"
  section: "38px"
components:
  button-primary:
    backgroundColor: "{colors.turf}"
    textColor: "{colors.canvas-white}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 22px"
    height: "52px"
  button-primary-hover:
    backgroundColor: "{colors.turf-dark}"
    textColor: "{colors.canvas-white}"
    rounded: "{rounded.control}"
  button-secondary:
    backgroundColor: "{colors.ink-navy}"
    textColor: "{colors.canvas-white}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "0 22px"
    height: "52px"
  input:
    backgroundColor: "{colors.canvas-white}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.control}"
    padding: "0 15px"
    height: "54px"
  card-team:
    backgroundColor: "{colors.quiet-surface}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.surface}"
  chip-limit:
    backgroundColor: "{colors.chip-surface}"
    textColor: "{colors.ink-soft}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "7px 10px"
---

# Design System: Footy AI

## Overview

**Creative North Star: "The Daylight Analyst Desk"**

Footy AI feels like a clean team sheet laid out on an analyst's desk: precise, calm, bright, and decisively football-specific. Ink navy supplies editorial authority, paper and white fields preserve daylight clarity, and turf green marks action without turning the interface into a stadium-themed novelty.

The visual system keeps model internals backstage. Progress is explicit, evidence is shown before a decision, and the interface asks users to identify teams through representative player imagery and plain-language names. Large headings establish the current task; compact match notation, fine rules, and restrained status color organize the details.

**Key Characteristics:**

- A split desktop workspace pairing a dark editorial introduction with a bright operating surface.
- Flat, ruled surfaces with generous whitespace and minimal ornamental depth.
- Player imagery as evidence, never color values as user-facing identity.
- Compact labels and match notation beneath bold, tightly tracked headlines.
- Turf green for action and state; signal lime for rare, high-contrast markers.

## Colors

The palette combines inky editorial neutrals with pitch-derived action colors and a sparse lime signal.

### Primary

- **Turf Green:** The main action and progress color for primary buttons, active tabs, completed steps, live status, and analysis indicators.
- **Deep Turf:** The stronger interaction state for button hover and low-emphasis text actions.

### Secondary

- **Signal Lime:** A rare, energetic marker used against ink navy for the brand glyph, active step markers, processing detail, and player placeholders.

### Neutral

- **Ink Navy:** The authoritative text, dark-panel, score-rule, icon-disc, and secondary-action color.
- **Soft Ink:** Supporting copy, metadata, inactive controls, and explanatory labels.
- **Paper:** The application ground and translucent masthead base.
- **Canvas White:** The main operating workspace, fields, and reversed text.
- **Quiet Surface:** A subtle off-white layer for drop zones and evidence cards.
- **Rule:** The shared divider and card-border color.
- **Intro Copy:** Muted light text that remains readable on the navy introduction.
- **Chip Surface:** The low-contrast fill for compact informational pills.
- **Field Border / Drop Border:** Slightly stronger neutral strokes reserved for editable controls and the upload target.
- **Disabled:** A muted action fill that communicates processing or temporary unavailability.
- **Error:** Reserved for actionable form and processing errors.

### Named Rules

**The Evidence Before Color Rule.** Team identity is communicated with a player image, a numbered kit label, and a team name; palette swatches and hex values never substitute for that evidence.

**The Rare Lime Rule.** Signal lime is a small, high-contrast annotation on navy or green structures, not a large surface color.

**The One Action Green Rule.** Turf green identifies the current actionable or progressing state. Avoid introducing competing accent hues.

## Typography

**Display Font:** Aptos (with Segoe UI and Arial fallbacks)
**Body Font:** Aptos (with Segoe UI and Arial fallbacks)
**Label Font:** Aptos (with Segoe UI and Arial fallbacks)

**Character:** A single contemporary sans-serif family keeps the product operational and familiar. Hierarchy comes from decisive scale, weight, tight headline tracking, and compact uppercase labels rather than a decorative type pairing.

### Hierarchy

- **Display:** Heavy, fluid, tightly tracked type with a near-solid line height. Use only for the introductory promise.
- **Headline:** Large, tightly tracked task titles for upload, progress, confirmation, and results views.
- **Title:** Bold supporting emphasis for upload prompts, team labels, player names, and action controls.
- **Body:** Calm explanatory copy with an open line height; helper text should stay near 66 characters per line where space allows.
- **Label:** Small, bold, uppercase notation with wide tracking for stage labels and result dividers.
- **Score Numerals:** Extra-heavy, tightly tracked numerals at a fluid display scale; keep the score visually central and unambiguous.

### Named Rules

**The Match Sheet Hierarchy Rule.** Use one strong headline, one compact uppercase stage label, and restrained supporting copy per view.

**The Numbers Stay Loud Rule.** Scores and jersey numbers may carry more weight than nearby prose because they are scan targets, not decoration.

## Layout

Desktop uses a two-column shell beneath a 72px masthead. The navy introduction occupies a minimum 310px and approximately 38% of the viewport; the white workspace takes the remaining width. Both sides use fluid clamp-based padding, while the active view is centered and capped at 760px.

The workflow is progressive rather than dashboard-like: upload, processing, team confirmation, and results occupy the same dominant workspace one at a time. Within a view, headings precede controls, actions sit at the foot of their content group, and rules separate match information without boxing every element.

At 900px and below, the shell becomes a single column, the three-step rail becomes horizontal, and workspace padding contracts. At 580px and below, candidate cards stack, headings become single-column groups, the masthead shortens to 62px, status copy collapses to its live dot, and result typography scales down while retaining the three-part score layout.

Spacing follows an 8–12–18–24–32px rhythm, with larger fluid gaps reserved for section breathing room. Preserve generous whitespace around the active task; do not densify the workspace into a control panel.

## Elevation & Depth

The system is flat by default. Depth comes from tonal contrast, one-pixel rules, dark/light field changes, and content clipping rather than floating cards. The only shadow is a soft green halo around the live service dot, which reads as status glow rather than elevation.

### Shadow Vocabulary

- **Live Status Halo** (`0 0 0 4px rgba(8,122,82,.12)`): Used only around the small online indicator.

### Named Rules

**The Flat Analyst Desk Rule.** Surfaces rest on the page plane; use rules and tonal shifts before adding shadows.

## Shapes

Forms are gently technical rather than bubbly. Standard controls use a 10px radius, evidence and upload surfaces use a 14px radius, and compact status elements use full pills or circles. The brand mark is the signature exception: three 8px corners and one clipped 2px corner create a precise flag-like silhouette.

Dashed borders identify the upload target, solid borders frame evidence and editable fields, and one-pixel rules organize steps and results. Player images are cropped into controlled rectangles with gently rounded corners; they should feel evidentiary, not like social-profile avatars.

**The Radius by Function Rule.** Use 14px for large bounded surfaces, 10px for controls and image thumbnails, and pills or circles only for compact status and progress markers.

## Components

### Buttons

- **Shape:** Sturdy rounded rectangle with a minimum 52px height and 10px corners.
- **Primary:** Turf green with white, extra-bold text and a balanced horizontal inset. A directional arrow may sit at the far side of the label.
- **Hover / Focus:** Hover deepens to Deep Turf and rises by 1px; keyboard focus uses a 3px translucent turf outline with a 3px offset. Disabled buttons mute to the disabled neutral, keep their size, and do not move.
- **Secondary:** Ink navy with white text, used for the annotated-video download.
- **Text:** Deep Turf text on no fill, reserved for low-emphasis restart actions.

### Chips

- **Style:** A compact quiet-neutral pill with soft-ink text, used for factual constraints such as the upload limit.
- **State:** Informational only; do not style it like a selectable filter.

### Cards / Containers

- **Corner Style:** Gently curved evidence surface using the larger 14px radius.
- **Background:** Quiet Surface over the white workspace.
- **Shadow Strategy:** None; a one-pixel rule supplies the boundary.
- **Border:** Solid neutral rule for candidate cards; dashed stronger neutral for the upload drop zone.
- **Internal Padding:** 18px around candidate inputs; the upload target uses a more generous 42px vertical inset.

### Inputs / Fields

- **Style:** White fill, field-neutral one-pixel stroke, 10px corners, 54px height, and 15px horizontal padding.
- **Focus:** A 3px translucent turf outline with a 3px offset, shared with buttons and tabs.
- **Error / Disabled:** Errors appear as direct red copy beneath the relevant form. Do not rely on color alone; the message carries the corrective instruction.

### Navigation

The masthead is a stable 72px identity rail with the brand at left and online status at right. Source tabs sit on a shared bottom rule; the selected tab gains a 3px inset turf underline and darker text. The three-step rail uses numbered circles, rules, and clear active/completed states; on narrow screens it becomes a horizontal three-column sequence.

### Team Evidence Card

Each detected team is represented by a large landscape player crop above a labeled team-name field. The image fills the card width, uses `object-fit: cover`, and favors the upper-center of the frame so the player remains legible. Pair cards in two columns when space allows and stack them on small screens.

### Score Sheet and Player Rows

The score sheet is a border-led composition rather than a filled card: team names anchor the edges and oversized numerals meet in the center. Player rows use a 64px evidence image or navy/lime placeholder, a name-and-team text group, and a right-aligned shirt number. Unavailable player identity remains explicit with “Name not confirmed” and an em dash rather than invented data.

## Do's and Don'ts

### Do:

- **Do** keep each workflow stage focused on one primary decision or reading task.
- **Do** show representative player imagery before asking users to name a detected team.
- **Do** preserve a clear hierarchy of stage label, task headline, helper copy, evidence, and action.
- **Do** use rules and tonal layering to organize score and evidence content.
- **Do** keep focus states visible and honor reduced-motion preferences.
- **Do** state unavailable player names or jersey numbers honestly.

### Don't:

- **Don't** expose hex values, cluster colors, or computer-vision vocabulary in the normal user journey.
- **Don't** identify teams through color alone.
- **Don't** add decorative shadows, gradients, glass effects, or stadium spectacle to routine surfaces.
- **Don't** use signal lime as a broad background or a competing primary action color.
- **Don't** crowd the workspace with simultaneous stages or dashboard panels.
- **Don't** guess player identity when OCR evidence is uncertain.

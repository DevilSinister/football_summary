# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is a football analyst or fan who wants a match summary without understanding computer-vision color values. This is inferred from the request's emphasis on a “normal user.”

## Product Purpose

Footy AI processes a football match video, detects the two playing teams from kit appearance, asks the user to name the detected teams using representative player images, and presents a readable score sheet with scorer team, shirt number, and player name when available.

## Positioning

The product turns raw match footage into named, player-level match results while keeping ambiguous team identity under human control.

## Operating Context

Users upload a match video or provide a direct video URL, wait while kit samples are detected, name each visual team, then review the processed result.

## Capabilities and Constraints

- Team colors are calculated from video frames and remain an internal implementation detail.
- The user confirms which detected kit belongs to each team by viewing a representative player crop.
- Jersey number and player name depend on OCR evidence and may be unavailable.
- Video processing uses local YOLO, event detection, and OCR models and can take several minutes.
- The product currently supports one active in-memory processing queue per backend process.

## Evidence on Hand

- Representative match footage: `input_videos/test.mp4`.
- Existing tracked-frame and player-crop examples: `output_videos/screenshot.png` and `output_videos/cropped_image.jpg`.
- Existing event, scorer, team, and persistence pipeline in the Python backend.

## Product Principles

- Ask users for football knowledge, never implementation values such as hex codes.
- Show the visual evidence behind a team assignment before asking for a name.
- Preserve uncertainty instead of inventing a player identity.
- Keep processing state and the next required action unmistakable.

## Accessibility & Inclusion

Color is never the only team identifier: every detected cluster includes a player image, a numbered label, and a text field.

# Frontend and backend branch handoff

This repository contains the Footy AI backend.

- **Backend source:** this repository, branch `jugaad`
- **Frontend source:** `https://github.com/DevilSinister/footyai.git`, branch `sinister/experimental`

Agents working on the frontend must pull the backend changes from the `jugaad`
branch of `football_summary`. Agents working on the backend must pull the frontend
changes from the `sinister/experimental` branch of `footyai`.

Always fetch before integration and keep changes scoped to the repository and branch
that own them.

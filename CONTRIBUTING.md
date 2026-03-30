## Contributing to AI Image Similar Search

Thank you for considering a contribution!

This project is a small prototype, so the process is intentionally lightweight. The goal is to keep the codebase approachable and easy to extend.

---

### Ways to contribute

- **Bug reports** – If something doesn’t work, please open an issue that includes:
  - Clear steps to reproduce.
  - Expected vs. actual behavior.
  - Environment details (OS, Python version, how you ran the app).
- **Feature ideas / improvements** – If you have an idea, open an issue first to discuss the approach.
- **Documentation** – Clarify or extend the README, comments, or example usage.
- **Code changes** – Fixes, refactors, or features that fit the scope of “small image search prototype”.

---

### Development setup

1. **Fork and clone** this repository.
2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the data** as described in `README.md` (at minimum, you’ll need `data/output/pdf_extracted_data.json` and sample images).

4. **Run the server** to verify everything works:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

---

### Coding guidelines

- **Clarity over cleverness**: Prefer straightforward, readable code.
- **Keep the surface small**: Avoid adding new top‑level scripts or entrypoints unless necessary.
- **Dependencies**:
  - Reuse existing libraries where possible.
  - If you must add a dependency, pick a well‑maintained, widely used library and explain the need in the pull request.
- **Error handling**:
  - Fail fast for clearly invalid inputs.
  - Log or surface helpful messages where appropriate.

---

### Pull request checklist

Before opening a PR:

- [ ] The app runs locally (`uvicorn main:app --reload`) without errors.
- [ ] Basic search still works end‑to‑end (upload an image on `http://localhost:8000/` and get results).
- [ ] New or changed behavior is documented in `README.md` (if user‑visible).
- [ ] No large, unrelated changes bundled in a single PR.

When you open the PR:

- Provide a short description of **what** you changed and **why**.
- Reference any related issues (e.g. `Fixes #123`).

---

### License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project (see `LICENSE`).


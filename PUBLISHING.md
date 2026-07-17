# Publishing checklist (for the authors)

Five steps take this folder to a citable archive. Delete this file (or keep
it — it's harmless) once done.

## 1. Create the GitHub repository and push

On github.com: **New repository** → name `slr-contaminant-screening`
(public, no README — this folder already has one). Then, from this folder:

```bash
git remote add origin git@github.com:<YOUR-USERNAME>/slr-contaminant-screening.git
git push -u origin main
```

(The folder is already a git repository with an initial commit. Use the
HTTPS URL instead of SSH if you prefer.)

## 2. Turn on the screening tool (GitHub Pages)

Repo → **Settings → Pages** → Source: *Deploy from a branch* →
Branch: `main`, folder: `/docs` → Save. After ~1 minute the tool is live at

```
https://<YOUR-USERNAME>.github.io/slr-contaminant-screening/
```

Paste that URL into the README where the Pages link placeholder is.

## 3. Mint a DOI (Zenodo)

1. Log in at zenodo.org with your GitHub account.
2. **GitHub** (under your account menu) → flip the toggle next to
   `slr-contaminant-screening`.
3. Back on GitHub: **Releases → Create a new release** → tag `v1.0.0`,
   title "v1.0.0 — JCH revision submission" → Publish.
4. Zenodo automatically archives the release and mints two DOIs: a
   *concept DOI* (always latest) and a *version DOI* (this release). The
   `.zenodo.json` in this repo supplies the metadata.
5. Add the DOI badge (Zenodo shows the markdown) to the README, replacing
   the placeholder comment.

## 4. Update the manuscript

Replace the "Software and data availability" statement with (fill in the
two URLs/DOI):

> **Software and data availability.** The full nonlinear Henry solver, all
> benchmark scripts and data, the figure-generation code, and the
> interactive screening calculator are openly available at
> https://github.com/YOUR-USERNAME/slr-contaminant-screening and archived
> at https://doi.org/10.5281/zenodo.XXXXXXX. The screening calculator can
> be used directly in a browser at
> https://YOUR-USERNAME.github.io/slr-contaminant-screening/.

The same URLs go in the response letter where the supplementary calculator
is mentioned.

## 5. Before tagging the release — two small TODOs

- README "Key numerical results" and figure outputs should match whatever
  final parameter choices survive co-author review; re-run `make all` if
  anything changed.
- If the paper's title/author list changes at acceptance, update
  `CITATION.cff` and `.zenodo.json`, then tag `v1.1.0` (Zenodo issues a
  fresh version DOI automatically).

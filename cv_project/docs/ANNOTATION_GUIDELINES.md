# Annotation Guidelines — Safety Helmet Detection

Tool: **Roboflow Annotate** · Export: **YOLOv8** · Classes (exact spelling):

| id | class | What to box |
|---|---|---|
| 0 | `helmet` | A safety helmet / hard hat **worn on a head**. Box the helmet plus the visible part of the head under it, from the helmet top to the chin/jaw line if visible. |
| 1 | `no_helmet` | A **bare head** — no safety helmet. Caps, hoods, turbans, scarves, hair, hats and headphones all count as `no_helmet`. Box the head from the top of the hair to the chin. |
| 2 | `person` | The **whole visible body** of a person, head to feet (or to the image edge). |

So a worker produces **two boxes**: one `person` box and one head-level box (`helmet` *or* `no_helmet`).

## Rules (from the task brief)
1. **Tight boxes** — edges touch the outermost visible pixels of the object; no gap, no cut-off.
2. **Label every visible object** of the three classes — missing labels teach the model that those objects are background.
3. **One box per object** — never one box around a group of people or heads.
4. **No excess background** inside a box.
5. **Do not guess** — if you cannot tell whether a head wears a helmet (too small, too blurry), leave it unlabelled rather than labelling randomly.
6. **Partially visible** objects: label only if the class is clear (e.g. half a helmet at the image edge is still clearly a helmet → label it).
7. A helmet **held in the hand or lying on the ground is NOT** `helmet` — the head wearing nothing is `no_helmet`.
8. **Consistent names** — only `helmet`, `no_helmet`, `person`; no variants such as `Helmet`, `head`, `no-helmet`.
9. **Review pass** — after finishing, go through all images once more and fix missed or loose boxes before exporting.

## Hard cases
| Situation | Decision |
|---|---|
| Baseball cap / cloth hat | `no_helmet` |
| Bicycle / motorcycle helmet on a head | `helmet` |
| Person seen from behind, helmet visible | `helmet` |
| Heavy occlusion: only a shoulder visible | `person` only if clearly a person; no head box |
| Reflections / posters of people | do not label |
| Very crowded, tiny heads (< ~8 px) | label only the ones you can classify with confidence |

## Quality evidence for the report
The notebook's **annotation audit** compares your boxes with the original public labels on the same images (agreement per class, extra objects you found, mean IoU = box tightness). Mention these numbers in the report's annotation section.

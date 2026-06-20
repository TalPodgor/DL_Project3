from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "project3_final_report.pdf"


def styles():
    base = getSampleStyleSheet()
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 10.4
    base["Normal"].leading = 13.2
    base["Normal"].alignment = TA_JUSTIFY
    base["Title"].fontName = "Helvetica-Bold"
    base["Title"].fontSize = 18
    base["Title"].leading = 22
    base["Title"].alignment = TA_CENTER
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 14
    base["Heading1"].leading = 17
    base["Heading1"].spaceBefore = 12
    base["Heading1"].spaceAfter = 6
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 11.5
    base["Heading2"].leading = 14
    base["Heading2"].spaceBefore = 8
    base["Heading2"].spaceAfter = 4
    base.add(ParagraphStyle(
        name="Subtitle",
        parent=base["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=13,
        spaceAfter=10,
    ))
    base.add(ParagraphStyle(
        name="Caption",
        parent=base["Normal"],
        fontSize=8.6,
        leading=10.5,
        textColor=colors.HexColor("#333333"),
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Small",
        parent=base["Normal"],
        fontSize=8.2,
        leading=10,
    ))
    return base


S = styles()


def p(text, style="Normal"):
    return Paragraph(text, S[style])


def h(text, level=1):
    return Paragraph(text, S["Heading1" if level == 1 else "Heading2"])


def bullet(items):
    return ListFlowable(
        [ListItem(p(x), leftIndent=10) for x in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontName="Helvetica",
        bulletFontSize=7,
    )


def table(data, widths=None, font_size=8.0, header=True):
    rows = []
    for row in data:
        rows.append([p(str(cell), "Small") for cell in row])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8b8b8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def fig(path, width, caption):
    img_path = ROOT / path
    parts = []
    if img_path.exists():
        im = Image(str(img_path))
        ratio = im.imageHeight / float(im.imageWidth)
        im.drawWidth = width
        im.drawHeight = width * ratio
        parts.append(im)
        parts.append(p(caption, "Caption"))
    else:
        parts.append(p(f"[Missing figure: {path}]", "Caption"))
    return KeepTogether(parts)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(A4[0] - 0.55 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.58 * inch,
    )
    story = []

    story += [
        p("Project 3: Synthetic-to-Real Chessboard Image Generation from FEN", "Title"),
        p("Final Report", "Subtitle"),
        p("Authors: Tal Podgor, Ilay Vanunu, Ran Packler &nbsp;&nbsp; | &nbsp;&nbsp; Course: Intro to Deep Learning, BGU", "Subtitle"),
        p("Repository: https://github.com/TalPodgor/DL_Project3", "Subtitle"),
        Spacer(1, 8),
        h("Abstract"),
        p(
            "This project addresses chessboard image generation from a FEN position and a requested "
            "camera viewpoint. The required output is a clean synthetic chessboard image and a more "
            "realistic image that preserves the same board state. We treat the problem as controlled "
            "synthetic-to-real image translation. Early unpaired translation with CUT/CycleGAN-style "
            "training produced realistic texture but did not reliably preserve exact board structure. "
            "The final system therefore uses a paired, pix2pixHD-style conditional generator, augmented "
            "with chess-specific semantic and geometric conditioning from Blender: an RGB render, a FEN "
            "semantic layout, a rendered silhouette, a depth-like map, and piece-edge cues. The selected "
            "checkpoint, chess_v5_bright_silABC, is trained on a camera-aligned Blender dataset with "
            "per-game oblique elevations of 38-44 degrees. On 140 held-out game-2 boards it achieves "
            "0.922 square accuracy, 0.992 occupancy accuracy, 0.804 occupied-piece type accuracy, and "
            "0.757 whole-board occupancy exactness under an independent square classifier. Qualitatively, "
            "the model gives the best practical trade-off between realistic wooden-board appearance and "
            "board-state preservation, although piece topology artifacts remain a central limitation."
        ),
    ]

    story += [
        h("1. Introduction"),
        p(
            "The task in Project 3 is to implement generate_chessboard_image(fen, viewpoint). Given a "
            "FEN string and a viewpoint string, the function must save synthetic.png, realistic.png, "
            "and side_by_side.png under ./results/. The generated realistic image should look like a "
            "photograph of a wooden chessboard while preserving the exact piece placement and the requested "
            "white-side or black-side view."
        ),
        p(
            "This task is harder than ordinary style transfer. A visually plausible chessboard is not "
            "sufficient: a pawn cannot appear on an empty square, a rook cannot become a bishop, and the "
            "camera orientation must remain consistent with the input viewpoint. At the same time, the real "
            "training photographs are low resolution, strongly rectified to the board plane, and contain "
            "small, low-contrast pieces. Tall pieces lean across square boundaries after rectification, so "
            "the board grid can be aligned while piece heads and bodies are not perfectly aligned."
        ),
        h("Main contributions", 2),
        bullet([
            "A FEN-controlled Blender rendering pipeline that creates synthetic chessboards and auxiliary control maps.",
            "A paired pix2pixHD-style translation model that conditions on RGB, semantic, depth, silhouette, and edge information.",
            "A camera-aligned V5 dataset that replaces near-top-down synthetic supervision with oblique Blender renders closer to the real photos.",
            "A set of quantitative and visual ablations showing why unpaired translation, silhouette-only conditioning, geometry locking, and naive parallax stretching were rejected.",
        ]),
    ]

    story += [
        h("2. Related Work"),
        p(
            "Image-to-image translation is commonly approached with conditional GANs. Pix2pix uses paired "
            "supervision and a conditional discriminator to learn a mapping from an input image to a target "
            "image. Pix2pixHD extends this idea with stronger high-resolution synthesis components, including "
            "multi-scale discrimination and feature matching. CycleGAN and CUT are unpaired approaches: they "
            "can learn domain style without one-to-one pairs, but their structure preservation is weaker for "
            "tasks that require exact object layout."
        ),
        p(
            "Our final method is closest to pix2pixHD: it uses paired supervision and a conditional generator, "
            "but adds chess-specific inputs rather than relying on RGB alone. CUT and CycleGAN-style training "
            "were useful early baselines, but they were not selected because the task is not only to make "
            "synthetic chessboards look real; it is to keep the exact chess position."
        ),
    ]

    story += [
        h("3. Data and Preprocessing"),
        p(
            "The final training set is datasets/chess_v5_oblique_aligned_bright. It contains paired images "
            "where the left half is the Blender-rendered source and the right half is the real target. Each "
            "sample also has a semantic silhouette image and a depth-like image. The split contains 736 "
            "training pairs from games 4, 5, 6, and 7, and 140 held-out test pairs from game 2. The held-out "
            "game split is important because it tests whether the model generalizes to unseen positions and "
            "viewpoint instances rather than memorizing one board sequence."
        ),
        p(
            "A central empirical finding was that camera geometry matters. The older V5 oblique dataset used "
            "a global elevation of approximately 54.9 degrees, while the real board photographs were more "
            "oblique. The final aligned-bright dataset uses per-game elevations of 38, 40, 42, and 44 degrees "
            "with a mean of 40.7 degrees. This reduces the mismatch between the synthetic piece projection "
            "and the real rectified photographs."
        ),
        table([
            ["Split", "Games", "Pairs", "Notes"],
            ["Train", "4, 5, 6, 7", "736", "Paired Blender source, real target, semantic silhouette, depth"],
            ["Test", "2", "140", "Held-out game used for quantitative and qualitative evaluation"],
        ], widths=[0.8 * inch, 1.3 * inch, 0.7 * inch, 3.7 * inch]),
        Spacer(1, 4),
        table([
            ["Dataset", "Camera elevation", "Resulting role"],
            ["chess_v5_oblique", "54.9 deg global", "Rejected as too top-down for real piece geometry"],
            ["chess_v5_oblique_aligned_bright", "38-44 deg per game", "Final training data for bright_silABC"],
        ], widths=[2.0 * inch, 1.5 * inch, 3.0 * inch]),
    ]

    story += [
        h("4. Method"),
        p(
            "The final model is called paired_geom_hd in the codebase. It is a paired conditional "
            "image-to-image model in the pix2pixHD family, not a CycleGAN. The generator is a ResNet "
            "9-block network with instance normalization. The discriminator is a multi-scale PatchGAN-style "
            "discriminator, and a local piece-crop discriminator is used during training to put additional "
            "pressure on occupied squares."
        ),
        h("Input representation", 2),
        p(
            "For each position, the model receives a tensor formed by concatenating: (1) the RGB synthetic "
            "render, (2) a one-hot semantic map, and (3) three geometry channels. The semantic map uses "
            "fen_silhouette mode: full-cell FEN information protects board occupancy, while rendered "
            "silhouette pixels provide a more accurate outline of visible piece shapes. The geometry channels "
            "are a depth-like render, a piece silhouette mask, and a silhouette boundary channel."
        ),
        h("Training losses", 2),
        p(
            "The generator is trained with adversarial loss, discriminator feature matching, VGG perceptual "
            "loss, masked L1 reconstruction, a silhouette-edge loss, contextual loss, a frozen square-classifier "
            "loss, and local crop-level piece losses. The masked L1 term uses a reduced weight on piece regions "
            "to avoid forcing pixel-perfect alignment where 3D parallax makes the real and synthetic piece "
            "heads differ. This is important because hard pixel losses on misaligned tall pieces tend to create "
            "blur or double-head artifacts."
        ),
        table([
            ["Parameter", "Final value"],
            ["Model", "paired_geom_hd"],
            ["Dataset mode", "v5_oblique"],
            ["Semantic source", "fen_silhouette"],
            ["Generator", "ResNet 9-blocks, ngf=64, instance norm"],
            ["Batch size", "1"],
            ["Epochs", "40 + 20 decay"],
            ["lambda_GAN, lambda_feat, lambda_VGG, lambda_L1", "1.0, 10.0, 3.0, 7.5"],
            ["l1_piece_w", "0.7"],
            ["lambda_edge, lambda_cx", "1.0, 0.5"],
            ["Local piece losses", "piece GAN=1.0, feature matching=10.0, VGG=5.0"],
        ], widths=[2.45 * inch, 4.05 * inch]),
        Spacer(1, 4),
        p(
            "At inference time, the model still satisfies the Project 3 contract: only the FEN and viewpoint "
            "are needed. The system renders the synthetic board from the FEN, derives the same control maps "
            "from that synthetic state, forwards the model, and saves the required PNG outputs."
        ),
    ]

    story += [
        fig("v5_work/report_final_qualitative.png", 5.6 * inch,
            "Figure 1. Final qualitative examples. Each row shows the Blender synthetic input, the selected bright_silABC output, and the real target image from the held-out test game."),
    ]

    story += [
        h("5. Experiments"),
        p(
            "We evaluate using both automatic and visual criteria. The main quantitative evaluator is an "
            "independent square classifier trained on real board crops. It reports square accuracy, occupancy "
            "accuracy, occupied-piece type accuracy, phantom and missing-piece rates, and whole-board exactness. "
            "These metrics are useful for board-state preservation, but they are not sufficient for final "
            "visual quality because a merged blob can sometimes be classified correctly. Therefore we also use "
            "visual grids and specialized audits for double-head, halo, transparency, and piece-detail artifacts."
        ),
        table([
            ["Run", "Square", "Occupancy", "Type", "Whole-board occupancy", "Phantom", "Missing"],
            ["bright_silAB", "0.9160", "0.9924", "0.7849", "0.6429", "0.0057", "0.0109"],
            ["bright_silABC (selected)", "0.9221", "0.9916", "0.8040", "0.7571", "0.0057", "0.0130"],
            ["silAB noPieceD", "0.9314", "0.9942", "0.8240", "0.7643", "0.0052", "0.0068"],
            ["pcomp_srcshape", "0.9235", "0.9955", "0.7979", "0.7929", "0.0019", "0.0090"],
            ["geometry lock train", "0.6993", "0.8738", "0.3075", "0.0000", "0.0293", "0.2991"],
            ["parallax8 fine-tune", "0.9094", "0.9846", "0.7879", "0.4500", "0.0167", "0.0130"],
        ], widths=[1.55 * inch, 0.72 * inch, 0.78 * inch, 0.62 * inch, 1.08 * inch, 0.7 * inch, 0.7 * inch]),
        p(
            "The selected model is not the winner of every scalar metric. For example, noPieceD has slightly "
            "higher classifier scores. However, final selection used a stricter practical criterion: the model "
            "must preserve a realistic appearance and not introduce severe synthetic, milky, or transparent "
            "piece artifacts. Bright_silABC was selected as the best overall visual trade-off."
        ),
    ]

    story += [
        h("6. Ablation Study"),
        p(
            "The final method contains several components, so we performed ablations and negative tests to "
            "justify the design. The most important conclusion is that improving board-state metrics alone is "
            "not enough. Components that force exact geometry can remove double heads, but often destroy the "
            "photo-like style. Components that preserve style often leave the topology problem unsolved."
        ),
        table([
            ["Ablation", "Hypothesis", "Observed result", "Decision"],
            ["Silhouette-only semantics", "Visible silhouettes should encode piece layout", "High phantom rate in the V5 probe; no whole-board occupancy exactness", "Reject; full-cell FEN is necessary"],
            ["Full-cell FEN + V5 geometry", "FEN layout protects occupancy while geometry provides shape", "Probe reached 1.0 occupancy on 16 boards", "Use as V5 baseline"],
            ["No piece classifier", "Classifier loss might overfit or reward blobs", "Improved some classifier metrics but did not clearly solve visual defects", "Not final"],
            ["Geometry lock", "Hard source geometry will prevent double heads", "Double-head defects dropped strongly, but pieces became washed and synthetic", "Reject for final style"],
            ["Two-stream / piece composite", "Separate foreground and background generation", "Style remained reasonable, topology mostly unchanged", "Not enough"],
            ["Parallax8 source stretch", "Synthetic pieces are still too short or top-down", "Worse occupancy, worse phantom rate, no double-head improvement", "Reject"],
        ], widths=[1.35 * inch, 1.75 * inch, 2.25 * inch, 1.15 * inch]),
        Spacer(1, 4),
        table([
            ["Run", "Double-head score", "Any-extra rate", "Pawns any-extra", "Officers any-extra"],
            ["bright_silAB", "2.4544", "0.5342", "0.5932", "0.4481"],
            ["bright_silABC", "2.4004", "0.5463", "0.6236", "0.4336"],
            ["parallax8", "2.5367", "0.5593", "0.6346", "0.4496"],
        ], widths=[1.6 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch]),
        fig("v5_work/report_ablation_qualitative.png", 6.0 * inch,
            "Figure 2. Qualitative ablation comparison. Geometry-lock improves structure but looks synthetic; parallax8 does not improve the selected bright_silABC model."),
    ]

    story += [
        h("7. What Did Not Work"),
        p(
            "Unpaired translation was the first major failure mode. CUT/CycleGAN-style models can transfer "
            "wooden-board style, but they do not receive a strong pairwise penalty for moving pieces or "
            "inventing piece-like texture on empty squares. This is unacceptable for chess, where the state "
            "must remain exact."
        ),
        p(
            "A second failure mode was naive geometric correction. The final aligned-bright dataset already "
            "uses camera elevations close to the real photos. A later parallax8 experiment stretched the "
            "synthetic piece silhouettes upward before fine-tuning from bright_silABC. It reduced neither "
            "the double-head audit nor the visual artifacts, and it degraded occupancy exactness from 0.757 "
            "to 0.450. This suggests that the remaining mismatch is not solved by a simple post-render stretch."
        ),
        p(
            "A third failure mode was hard geometry locking. It made double heads much rarer, but it achieved "
            "this by injecting synthetic-looking piece structure into the output. The resulting pieces looked "
            "washed, transparent, or milky, which failed the realistic-image requirement."
        ),
        p(
            "Finally, automatic mask extraction with SAM and color trimming did not produce clean real-piece "
            "sprites. At the available 40-60 pixel piece scale, many white-piece pixels have almost the same "
            "color as light board squares. Automatic segmentation often captured the square or fragmented the "
            "piece, so it was not used in the final model."
        ),
    ]

    story += [
        h("8. Discussion and Limitations"),
        p(
            "The selected model is a practical submission model, not a complete solution to realistic chess "
            "piece synthesis. It generally preserves occupancy well and creates a visually plausible wooden "
            "board, but local piece geometry remains imperfect. Common failures include blob-like pawns, "
            "side-lobed heads, weak separation between neighboring pieces, and occasional phantom texture on "
            "empty squares."
        ),
        p(
            "The main limitation is the supervision itself. The real targets are board-rectified photographs "
            "of a 3D scene. A homography aligns the board plane, but not the elevated heads of pieces. Pixel "
            "and perceptual losses therefore supervise slightly incompatible shapes. Reducing L1 pressure on "
            "piece regions avoids severe blur, but also leaves the generator with weaker direct supervision "
            "for fine piece structure."
        ),
        p(
            "The most credible future improvement would be new supervision rather than more loss tuning: "
            "manual annotation of real piece masks, higher-resolution capture, or a renderer/camera calibration "
            "pipeline that matches the real camera pose and lens more precisely. With such data, one could train "
            "a dedicated foreground matting or piece-refinement model instead of asking a small GAN to infer "
            "ambiguous local topology from low-resolution photos."
        ),
    ]

    story += [
        h("9. Conclusion"),
        p(
            "We built a controlled synthetic-to-real chessboard generation pipeline for Project 3. The final "
            "model is a paired pix2pixHD-style generator with FEN, silhouette, depth, and edge conditioning. "
            "The selected bright_silABC checkpoint is trained on the camera-aligned bright V5 dataset and gives "
            "the best overall trade-off between realistic appearance and state preservation among the tested "
            "models. The project also produced a clear negative result: under the current low-resolution data "
            "and automatic supervision, there is a persistent trade-off between clean piece topology and "
            "realistic photographic style. This limitation is documented through both quantitative ablations "
            "and visual comparisons."
        ),
    ]

    story += [
        h("References"),
        p("[1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. CVPR, 2017."),
        p("[2] T.-C. Wang et al. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs. CVPR, 2018."),
        p("[3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV, 2017."),
        p("[4] T. Park, A. A. Efros, R. Zhang, and J.-Y. Zhu. Contrastive Learning for Unpaired Image-to-Image Translation. ECCV, 2020."),
        p("[5] N. Ravi et al. Segment Anything. ICCV, 2023."),
        p("[6] Blender Foundation. Blender, open-source 3D creation suite. https://www.blender.org/"),
        p("[7] Project repository: https://github.com/TalPodgor/DL_Project3"),
    ]

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    build()

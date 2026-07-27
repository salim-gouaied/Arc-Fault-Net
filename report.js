const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageBreak, LevelFormat, TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Color palette ──────────────────────────────────────────────
const C = {
  blue:       "1D4ED8",
  blue_light: "DBEAFE",
  blue_mid:   "3B82F6",
  red:        "DC2626",
  red_light:  "FEE2E2",
  green:      "15803D",
  green_light:"DCFCE7",
  amber:      "B45309",
  amber_light:"FEF3C7",
  purple:     "6D28D9",
  purple_light:"EDE9FE",
  slate:      "334155",
  slate_light:"F1F5F9",
  slate_mid:  "E2E8F0",
  slate_border:"CBD5E1",
  white:      "FFFFFF",
  text:       "0F172A",
  muted:      "64748B",
  heading_bg: "1E3A5F",
};

// ── Helpers ────────────────────────────────────────────────────
const b  = (str, color) => new TextRun({ text: str, bold: true, color: color || C.text, font: "Arial" });
const t  = (str, color, size) => new TextRun({ text: str, color: color || C.text, font: "Arial", size: size || 20 });
const mono = (str, color) => new TextRun({ text: str, font: "Courier New", color: color || C.blue, size: 18, bold: true });
const sp = () => new TextRun({ text: " ", font: "Arial" });

const cellBorder = (color) => {
  const s = { style: BorderStyle.SINGLE, size: 6, color: color || C.slate_border };
  return { top: s, bottom: s, left: s, right: s };
};
const noBorder = () => {
  const s = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return { top: s, bottom: s, left: s, right: s };
};

const para = (children, opts = {}) => new Paragraph({
  children,
  spacing: { before: opts.before ?? 40, after: opts.after ?? 40 },
  alignment: opts.align || AlignmentType.LEFT,
  indent: opts.indent ? { left: opts.indent } : undefined,
  ...opts.extra
});

const heading1 = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, color: C.white, font: "Arial", size: 32 })],
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 300, after: 120 },
  shading: { fill: C.heading_bg, type: ShadingType.CLEAR },
  indent: { left: 160, right: 160 },
});

const heading2 = (text, color) => new Paragraph({
  children: [new TextRun({ text, bold: true, color: color || C.blue, font: "Arial", size: 26 })],
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 80 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: color || C.blue } },
});

const heading3 = (text, color) => new Paragraph({
  children: [new TextRun({ text, bold: true, color: color || C.slate, font: "Arial", size: 22 })],
  spacing: { before: 160, after: 60 },
});

const rule = (color) => new Paragraph({
  children: [new TextRun({ text: "", font: "Arial" })],
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: color || C.slate_border } },
  spacing: { before: 80, after: 80 },
});

const labelBadge = (label, fill, textColor) => new TextRun({
  text: `  ${label}  `,
  bold: true, font: "Arial", size: 16,
  color: textColor || C.white,
  shading: { fill, type: ShadingType.CLEAR },
});

// Simple coloured info box via a single-cell table
const infoBox = (children, fill, borderColor, w) => new Table({
  width: { size: w || 9360, type: WidthType.DXA },
  columnWidths: [w || 9360],
  borders: { ...(() => { const s = { style: BorderStyle.SINGLE, size: 8, color: borderColor || C.blue }; return { top: s, bottom: s, left: s, right: s }; })() },
  rows: [new TableRow({
    children: [new TableCell({
      borders: cellBorder(borderColor || C.blue),
      shading: { fill: fill || C.blue_light, type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 160, right: 160 },
      width: { size: w || 9360, type: WidthType.DXA },
      children,
    })]
  })]
});

// Two-column table row helper
const twoCol = (left, right, lw, rw, lFill, rFill) => new TableRow({
  children: [
    new TableCell({
      borders: cellBorder(C.slate_border),
      shading: { fill: lFill || C.white, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 140, right: 140 },
      width: { size: lw, type: WidthType.DXA },
      children: left,
    }),
    new TableCell({
      borders: cellBorder(C.slate_border),
      shading: { fill: rFill || C.white, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 140, right: 140 },
      width: { size: rw, type: WidthType.DXA },
      children: right,
    }),
  ]
});

// Header row for tables
const headerRow = (cols, widths, fill) => new TableRow({
  tableHeader: true,
  children: cols.map((c, i) => new TableCell({
    borders: cellBorder(C.slate_border),
    shading: { fill: fill || C.heading_bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    children: [para([new TextRun({ text: c, bold: true, color: C.white, font: "Arial", size: 18 })],
      { before: 0, after: 0 })],
  }))
});

const dataRow = (cols, widths, fills, bold) => new TableRow({
  children: cols.map((c, i) => new TableCell({
    borders: cellBorder(C.slate_border),
    shading: { fill: fills?.[i] || C.white, type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 120, right: 120 },
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    children: [para([new TextRun({
      text: c, font: "Arial", size: 18,
      color: bold?.[i] ? C.text : C.muted,
      bold: !!bold?.[i]
    })], { before: 0, after: 0 })],
  }))
});

// ── bullet list helper ─────────────────────────────────────────
const bullet = (text, indent, color) => new Paragraph({
  numbering: { reference: "bullets", level: indent || 0 },
  children: [new TextRun({ text, font: "Arial", size: 20, color: color || C.text })],
  spacing: { before: 30, after: 30 },
});

const bullet2 = (parts) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  children: parts,
  spacing: { before: 30, after: 30 },
});

// ── DOCUMENT ──────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 260 } } }
        }, {
          level: 1, format: LevelFormat.BULLET, text: "\u25E6",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 900, hanging: 260 } } }
        }]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 20, color: C.text } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: C.white },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: C.blue },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    children: [

      // ══════════════════════════════════════════════════════════
      // COVER BLOCK
      // ══════════════════════════════════════════════════════════
      new Paragraph({
        children: [new TextRun({ text: "ARC-FaultNet V2", bold: true, font: "Arial", size: 52, color: C.heading_bg })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 600, after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Architecture Report — Slides Content Brief", font: "Arial", size: 28, color: C.muted })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Channel Attention  \u2022  Load Generalization  \u2022  Cross-Branch Fusion", font: "Arial", size: 22, color: C.blue })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 400 },
      }),
      rule(C.blue),

      // ══════════════════════════════════════════════════════════
      // SECTION 1 — CHANNEL ATTENTION
      // ══════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading1("SECTION 1 — Channel Attention: Physical Justification"),

      para([t("The model receives "), b("4 physically distinct channels"), t(" derived entirely from "), mono("I(t)"),
        t(". Voltage "), mono("V(t)"), t(" is "), b("excluded from the model"), t(" — it is used only for zero-crossing cycle segmentation outside the learning pipeline.")]),

      // ── Why not V(t)? ─────────────────────────────────────────
      heading2("1.1  Why V(t) Is Excluded from Model Input"),
      para([t("Three reasons eliminate V(t) as a learning channel:")]),
      bullet2([b("Resistive loads: ", C.red), t("V(t) = R · I(t). The model would see a scaled copy of I(t) — zero new information.")]),
      bullet2([b("Inductive/SMPS loads: ", C.red), t("The phase shift between V and I encodes the load type, not the arc. The model risks learning load identity instead of arc signature, which destroys generalization to unknown appliances.")]),
      bullet2([b("Arc physics: ", C.red), t("All arc signatures — flat shoulder, HF burst, amplitude depression — are exclusively observable in I(t). The voltage changes are secondary and less reliable across load types.")]),

      // ── 4 Channels Table ──────────────────────────────────────
      heading2("1.2  The Four Input Channels — Design Rationale"),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [560, 1800, 3000, 2000, 2000],
        rows: [
          headerRow(["", "Channel", "Formula / Definition", "Physical Meaning", "Dominant for Load"], [560, 1800, 3000, 2000, 2000]),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.blue), shading: { fill: C.blue, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 100, right: 100 }, width: { size: 560, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [para([new TextRun({ text: "C1", bold: true, color: C.white, font: "Arial", size: 20 })], { before: 0, after: 0, align: AlignmentType.CENTER })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.blue_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("I(t) Raw", C.blue), t("\nRaw waveform", C.muted, 17)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [para([mono("x(t) = I(t)")], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Global shape, amplitude, harmonic content. Anchor channel — always informative.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([b("All types", C.muted)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.red), shading: { fill: C.red, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 100, right: 100 }, width: { size: 560, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [para([new TextRun({ text: "C2", bold: true, color: C.white, font: "Arial", size: 20 })], { before: 0, after: 0, align: AlignmentType.CENTER })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.red_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("Dowalla Residual", C.red), t("\nInter-cycle delta", C.muted, 17)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [para([mono("residual_k = I_k \u2212 I_{k\u22121}")], { before: 0, after: 0 }), para([t("Reconstructed as 1D signal by concatenating N-1 cycle residuals", C.muted, 17)], { before: 20, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("~ 0 when no arc fires. Reveals flat shoulder as negative trough (resistive) or isolates spike (SMPS). Automatically cancels stable load signature.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([b("SMPS + Multi-load", C.red)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.green), shading: { fill: C.green, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 100, right: 100 }, width: { size: 560, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [para([new TextRun({ text: "C3", bold: true, color: C.white, font: "Arial", size: 20 })], { before: 0, after: 0, align: AlignmentType.CENTER })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("TKEO", C.green), t("\nInstantaneous energy", C.muted, 17)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [para([mono("I[n]\u00B2 \u2212 I[n\u22121] \u00B7 I[n+1]")], { before: 0, after: 0 }), para([t("Teager-Kaiser Energy Operator", C.muted, 17)], { before: 20, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Sensitive to AM+FM modulation. Arc ignition/extinction = sub-cycle energy burst. Millisecond precision, better than STFT window resolution.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([b("Inductive / Motor", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.amber), shading: { fill: C.amber, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 100, right: 100 }, width: { size: 560, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [para([new TextRun({ text: "C4", bold: true, color: C.white, font: "Arial", size: 20 })], { before: 0, after: 0, align: AlignmentType.CENTER })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.amber_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("RMS Sliding", C.amber), t("\nAmplitude envelope", C.muted, 17)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [para([mono("RMS over sliding window M/4")], { before: 0, after: 0 }), para([t("Local RMS computed every M/8 samples", C.muted, 17)], { before: 20, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Reveals amplitude depression: series arc reduces current. Flat shoulder visible as envelope collapse near zero-crossing.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([b("Resistive loads", C.amber)], { before: 0, after: 0 })] }),
          ]}),
        ]
      }),

      // ── C2 vs dI/dt ──────────────────────────────────────────
      heading2("1.3  Why C2 Is the Dowalla Residual and Not |dI/dt|"),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.red), shading: { fill: C.red_light, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 160, right: 160 }, width: { size: 4680, type: WidthType.DXA }, children: [
              para([new TextRun({ text: "\u2718  Rejected: |dI/dt| = |I[n] \u2212 I[n\u22121]|", bold: true, color: C.red, font: "Arial", size: 22 })], { before: 0, after: 60 }),
              bullet2([b("Problem: ", C.red), t("The derivative is maximum at the zero-crossing of a sinusoid (where cos is maximum) and minimum at the peaks. This is the natural behaviour of a pure 50Hz signal.")]),
              bullet2([b("Consequence: ", C.red), t("The flat shoulder effect — the primary arc signature for resistive loads — occurs near the zero-crossing, exactly where |dI/dt| is already naturally large. The arc signal is buried in background noise.")]),
              bullet2([b("Result: ", C.red), t("Poor signal-to-noise ratio for the most important arc signature. The channel does not discriminate between arc and non-arc conditions near zero-crossing.")]),
            ]}),
            new TableCell({ borders: cellBorder(C.green), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 160, right: 160 }, width: { size: 4680, type: WidthType.DXA }, children: [
              para([new TextRun({ text: "\u2714  Chosen: residual_k = I_k \u2212 I_{k\u22121}", bold: true, color: C.green, font: "Arial", size: 22 })], { before: 0, after: 60 }),
              bullet2([b("Flat background: ", C.green), t("The residual is approximately zero everywhere on a clean periodic signal. The baseline is flat, not oscillating with the sinusoid.")]),
              bullet2([b("High contrast: ", C.green), t("Any non-zero value means something changed between cycle k and cycle k\u22121. Arc uniquely produces this change. The flat shoulder appears as a negative trough; SMPS arc as an isolated spike.")]),
              bullet2([b("Load self-cancellation: ", C.green), t("Stable load signatures (SMPS switching, motor commutation) repeat identically each cycle and subtract to zero. Only the arc — being aperiodic — remains.")]),
              bullet2([b("Morphology preservation: ", C.green), t("Unlike scalar descriptors (E_mod, ED, MSSD), the full vector form preserves the shape, width, and phase location of the arc perturbation within the cycle.")]),
            ]}),
          ]}),
        ]
      }),

      // ── Attention mechanism ───────────────────────────────────
      heading2("1.4  Channel Attention Mechanism — How the Weights Are Computed"),

      infoBox([
        para([b("Formula: ", C.blue), mono("\u03B2 = \u03C3( MLP(AvgPool(F)) + MLP(MaxPool(F)) )  \u2208 (0,1)\u2074")], { before: 0, after: 40 }),
        para([t("One scalar weight \u03B2_c is computed per channel. It answers the question: "), b("\"for this specific window, which channel carries the most arc-discriminant information?\""), t(" The answer is load-dependent and learned from data — no prior knowledge of load type is required.")], { before: 0, after: 0 }),
      ], C.blue_light, C.blue),

      para([t("The mechanism processes each of the 4 channels as follows:")]),
      bullet2([b("Global Average Pool: ", C.blue), t("compresses the channel to a single scalar representing its mean energy level.")]),
      bullet2([b("Global Max Pool: ", C.blue), t("captures the peak activation, sensitive to sparse high-energy events like arc spikes.")]),
      bullet2([b("Shared MLP (4 \u2192 16 \u2192 4): ", C.blue), t("learns non-linear interactions between channels — for example, that high C2 residual combined with low C4 RMS suggests SMPS arc rather than resistive arc.")]),
      bullet2([b("Sigmoid: ", C.blue), t("produces weights in (0,1) that gate each channel. Low-weight channels are suppressed; high-weight channels are emphasized.")]),

      // ── Load specialization table ─────────────────────────────
      heading2("1.5  Expected Channel Weight Specialization per Load Type"),

      para([t("After training, the channel attention is expected to specialize as follows. This specialization is learned from data, not hardcoded — the model adapts at inference time based on the signal context alone:")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1500, 1800, 900, 900, 900, 900, 1460],
        rows: [
          headerRow(["Load Type", "Arc Signature", "C1\n(Raw)", "C2\n(Residual)", "C3\n(TKEO)", "C4\n(RMS)", "Dominant"], [1500, 1800, 900, 900, 900, 900, 1460]),
          dataRow(["Resistive\n(Bulb, Furnace)", "Flat shoulder\nnear zero-crossing", "Low", "Medium", "Low", "HIGH", "C4 RMS Envelope"], [1500, 1800, 900, 900, 900, 900, 1460], [C.white, C.white, C.white, C.white, C.white, C.white, C.amber_light], [1,0,0,0,0,0,1]),
          dataRow(["SMPS\n(PC, TV)", "Narrow spike at\nrandom phase", "Medium", "HIGH", "Medium", "Low", "C2 Residual"], [1500, 1800, 900, 900, 900, 900, 1460], [C.white, C.white, C.white, C.white, C.white, C.white, C.red_light], [1,0,0,0,0,0,1]),
          dataRow(["Inductive\n(Motor, Vacuum)", "Commutation\n+ arc burst", "Low", "Low", "HIGH", "Low", "C3 TKEO Energy"], [1500, 1800, 900, 900, 900, 900, 1460], [C.white, C.white, C.white, C.white, C.white, C.white, C.green_light], [1,0,0,0,0,0,1]),
          dataRow(["Multi-Load\n(6 devices masked)", "Arc signal diluted\nby masking loads", "Low", "HIGH", "Low", "Low", "C2 Residual\n(self-cancels others)"], [1500, 1800, 900, 900, 900, 900, 1460], [C.white, C.white, C.white, C.white, C.white, C.white, C.red_light], [1,0,0,0,0,0,1]),
        ]
      }),

      // ── Generalization table ──────────────────────────────────
      heading2("1.6  Load-Agnostic Generalization — Challenge \u2192 Mechanism Mapping"),

      para([t("The architecture detects arcs without prior knowledge of load type. The following mechanisms provide implicit load adaptation at inference time:")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2200, 4600, 1200, 1360],
        rows: [
          headerRow(["Challenge", "Resolving Mechanism", "Key Feature", "Result"], [2200, 4600, 1200, 1360]),
          dataRow(["Unknown load type\n(never seen in training)", "C2 residual self-cancels any stable periodic signature. Channel attention auto-weights the most informative channel.", "C2 + CAM", "\u2714 Generalizes"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
          dataRow(["Multi-load masking\n(arc diluted 1/6)", "C2 dominant: cancels non-arcing load signatures. CRC normalizes arc perturbation by local RMS. BiGRU accumulates evidence over 50 cycles.", "C2 + CRC\n+ BiGRU", "\u2714 Robust"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
          dataRow(["Motor brush noise\n(vacuum cleaner FP)", "BiGRU detects strict periodicity pattern. C2 subtracts repeatable commutation events (same phase each cycle).", "C2 + BiGRU\nhidden state", "\u2714 FP Suppressed"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
          dataRow(["Low-power arc\n(60W bulb)", "CRC = MSSD/RMS_{k-1} normalizes perturbation by load amplitude. C4 envelope sensitive to small depressions. ZCP descriptor.", "CRC + C4\n+ ZCP", "\u2714 Detectable"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
          dataRow(["Intermittent arc\n7/50 cycles only", "BiGRU cycle attention weights the 7 arcing tokens. XGBoost embedding shaped by those 7 cycles, ignores 43 silent ones. IEC 62606 equivalent.", "BiGRU\nCycle Attention", "\u2714 IEC-compliant"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
          dataRow(["Variable load power\n(50W PC vs 1100W furnace)", "Per-cycle RMS normalization at input stage. CRC descriptor provides load-power-invariant perturbation score.", "RMS norm\n+ CRC", "\u2714 Scale-invariant"], [2200, 4600, 1200, 1360], [C.white, C.white, C.white, C.green_light], [1,0,1,1]),
        ]
      }),

      // ══════════════════════════════════════════════════════════
      // SECTION 2 — CROSS-BRANCH FUSION
      // ══════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading1("SECTION 2 — Cross-Branch Fusion and Robustness"),

      // ── Two phenomena ─────────────────────────────────────────
      heading2("2.1  Why Two Branches Are Physically Necessary"),

      para([t("The series arc produces "), b("two physically distinct phenomena"), t(" that occur at entirely different timescales. Neither branch alone can observe both.")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.blue), shading: { fill: C.blue_light, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 160, right: 160 }, width: { size: 4680, type: WidthType.DXA }, children: [
              para([new TextRun({ text: "\u23F1  Phenomenon 1 — Inter-Cycle Discontinuity", bold: true, color: C.blue, font: "Arial", size: 24 })], { before: 0, after: 80 }),
              para([b("Timescale: ", C.blue), t("~20 ms (one 50Hz cycle)")], { before: 0, after: 30 }),
              para([b("What happens: ", C.blue), t("The arc interrupts current flow intermittently. The waveform shape of cycle k differs from cycle k\u22121 at the phase where the arc fired. Across 50 cycles, this produces an irregular pattern of perturbed vs clean cycles.")], { before: 0, after: 30 }),
              para([b("Captured by: ", C.blue), t("Temporal branch (Branch A Dowalla scalars + Branch B Conv1D on residual + BiGRU temporal memory)")], { before: 0, after: 30 }),
              para([b("Blind to: ", C.blue), t("Fast sub-millisecond transients within a single cycle — these are averaged out by cycle-level processing")], { before: 0, after: 0 }),
            ]}),
            new TableCell({ borders: cellBorder(C.amber), shading: { fill: C.amber_light, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 160, right: 160 }, width: { size: 4680, type: WidthType.DXA }, children: [
              para([new TextRun({ text: "\u26A1  Phenomenon 2 — Intra-Cycle HF Burst", bold: true, color: C.amber, font: "Arial", size: 24 })], { before: 0, after: 80 }),
              para([b("Timescale: ", C.amber), t("Sub-millisecond to a few ms")], { before: 0, after: 30 }),
              para([b("What happens: ", C.amber), t("Arc plasma ignition and extinction injects a broadband high-frequency burst (kHz range) onto the fundamental sinusoid within a single cycle. This burst lasts microseconds to milliseconds and is visible in the spectrogram as a sudden column of elevated energy across multiple frequency bins.")], { before: 0, after: 30 }),
              para([b("Captured by: ", C.amber), t("Spectral branch (Branch C — STFT with FreqGate and asymmetric pooling)")], { before: 0, after: 30 }),
              para([b("Blind to: ", C.amber), t("Slow inter-cycle shape changes — STFT within a single cycle window cannot see how adjacent cycles differ")], { before: 0, after: 0 }),
            ]}),
          ]}),
        ]
      }),

      infoBox([
        para([new TextRun({ text: "Key Insight: ", bold: true, color: C.green, font: "Arial", size: 20 }), t("These are two manifestations of the same arc event at different timescales. The empirical accuracy improvement observed when adding the spectral branch confirms that the two branches are capturing genuinely complementary physical information — not two representations of the same thing.", C.text, 20)], { before: 0, after: 0 }),
      ], C.green_light, C.green),

      // ── V1 CAM bug ────────────────────────────────────────────
      heading2("2.2  V1 Architecture Bug: The Channel-Ordering Problem in CAM"),

      para([t("The original V1 architecture concatenated the two branch outputs and split the CAM weights as:"), mono(" cam_w[:C]"), t(" for temporal and "), mono("cam_w[C:]"), t(" for spectral. This is incorrect.")]),

      infoBox([
        para([b("The Bug: ", C.red), t("The CAM processes the joint concatenated tensor (batch, 2C). After training, the network mixes information freely across both branches inside the CAM. There is no guarantee that trained channels 0–127 remain \"temporal\" and 128–255 remain \"spectral\". The split is arbitrary and uninterpretable after training.")], { before: 0, after: 40 }),
        para([b("Consequence: ", C.red), t("The attention weights assigned to one branch may in reality reflect the other branch's content. Visualization of cam_L and cam_H becomes meaningless. The cross-branch guidance intended by the design does not function as documented.")], { before: 0, after: 0 }),
      ], C.red_light, C.red),

      // ── Fixed CAM ────────────────────────────────────────────
      heading2("2.3  V2 Fix: Conditioned Cross-Branch Gates"),

      para([t("The V2 design replaces the split-CAM with two "), b("independent conditioned gates"), t(", each of which explicitly receives context from "), b("both branches"), t(". This makes the cross-branch guidance architecturally traceable and semantically correct.")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2200, 7160],
        rows: [
          headerRow(["Component", "Specification and Behaviour"], [2200, 7160], C.purple),
          dataRow(["Input", "f_temporal (B,128) from BiGRU context vector\nf_spectral (B,128) from spectral Branch C global average pool"], [2200, 7160], [C.white, C.white], [1,0]),
          dataRow(["Joint Context", "joint = concat([f_temporal, f_spectral])  \u2208  (B, 256)"], [2200, 7160], [C.white, C.purple_light], [1,1]),
          dataRow(["Temporal Gate", "gate_T = \u03C3( MLP([f_T \u2225 f_S]) )  \u2208 (B,128)\nTemporal gate sees spectral context \u2192 if spectral branch shows no HF burst, gate_T suppresses temporal evidence (FP prevention for motor loads)"], [2200, 7160], [C.white, C.white], [1,0]),
          dataRow(["Spectral Gate", "gate_S = \u03C3( MLP([f_T \u2225 f_S]) )  \u2208 (B,128)\nSpectral gate sees temporal context \u2192 if BiGRU already identified a firing pattern, gate_S amplifies spectral HF burst evidence"], [2200, 7160], [C.white, C.white], [1,0]),
          dataRow(["Gated Features", "f_T_gated = f_temporal \u2299 gate_T\nf_S_gated = f_spectral \u2299 gate_S"], [2200, 7160], [C.white, C.white], [1,0]),
          dataRow(["Final Fusion", "f_fused = Linear(256\u2192128)( concat([f_T_gated, f_S_gated]) )\n\u2192 128-dimensional embedding, input to XGBoost"], [2200, 7160], [C.white, C.green_light], [1,1]),
        ]
      }),

      // ── Scenario robustness ───────────────────────────────────
      heading2("2.4  Architecture Behaviour Under Diverse Operating Conditions"),

      para([t("The following describes how each component combination handles the key operating scenarios encountered in real household electrical networks:")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1800, 2000, 3760, 1800],
        rows: [
          headerRow(["Scenario", "Arc Signature", "Architecture Response", "Expected Outcome"], [1800, 2000, 3760, 1800]),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\uD83C\uDFE0 Single Resistive\n(Bulb, Furnace)", C.green)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Flat shoulder near zero-crossing. Amplitude depression.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("C4 (RMS envelope) ", C.amber), t("dominant — flat shoulder visible as amplitude collapse near zero-crossing. ZCP descriptor spikes in Branch A. Spectral branch detects HF plasma burst if fs sufficient. BiGRU confirms repeated firing pattern across cycles.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("High confidence detection", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.blue_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\uD83D\uDCBB SMPS Load\n(PC, TV)", C.blue)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Narrow impulsive spike at random phase. Normal SMPS switching already creates HF harmonics.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("C2 residual ", C.red), t("cancels normal SMPS switching pattern (identical each cycle). Remaining residual isolates the arc spike at its random phase angle. Spectral branch detects broadband burst above the switching harmonic peaks.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("Spike isolated cleanly", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.amber_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\uD83D\uDD0C 6-Device Multi-Load\n(masked scenario)", C.amber)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Arc signal from one device diluted to ~1/6 of total current. SNR severely degraded.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("C2 residual ", C.red), t("self-cancels the 5 non-arcing loads. "), b("CRC ", C.blue), t("normalizes the arc perturbation by local RMS. "), b("BiGRU ", C.green), t("accumulates weak evidence over 50 cycles — even a 1/6-amplitude signal repeated across 7+ cycles generates a recognizable embedding.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("SNR recovery via temporal aggregation", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.red_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\u2699\uFE0F Motor Load\n(Vacuum cleaner)", C.red)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Motor brush naturally generates micro-arcs at commutation. Primary false positive source per Dowalla.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("C2 residual ", C.red), t("subtracts repeatable commutation events (same phase each cycle). "), b("BiGRU ", C.green), t("encodes strict periodicity in hidden state — recognizable as non-fault pattern. "), b("Cross-gate: ", C.purple), t("spectral branch confirms absence of random-phase HF burst \u2192 gate_T suppresses temporal arc evidence.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("False positives suppressed", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\u26A1 Intermittent Arc\n7 / 50 cycles", C.green)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("IEC 62606:2013 minimum threshold. Arc fires in only 7 of 50 cycles, non-consecutively.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("BiGRU cycle attention ", C.green), t("assigns high weight to the 7 arcing tokens during training. The final context vector is disproportionately shaped by these 7 cycles. XGBoost receives an embedding that reflects the arc signature even though 43/50 cycles were silent.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("IEC 62606 compliant", C.green)], { before: 0, after: 0 })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.blue_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("\uD83D\uDCCA Unknown Load\n(not in training set)", C.blue)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2000, type: WidthType.DXA }, children: [para([t("Appliance signature unknown. Model has never seen this load type during training.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.white, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3760, type: WidthType.DXA }, children: [para([b("C2 residual ", C.red), t("cancels any stable periodic signature regardless of its source — no knowledge of load type required. "), b("CRC ", C.blue), t("normalizes by the load's own RMS. Channel attention auto-weights based on which channel shows elevated values. No retraining required for new appliance types.", C.muted, 18)], { before: 0, after: 0 })] }),
            new TableCell({ borders: cellBorder(C.slate_border), shading: { fill: C.green_light, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [para([b("Zero-shot generalization", C.green)], { before: 0, after: 0 })] }),
          ]}),
        ]
      }),

      // ── BiGRU vs GAP ─────────────────────────────────────────
      heading2("2.5  Why BiGRU Instead of Global Average Pooling"),

      para([t("A naive architecture using Global Average Pooling (GAP) over the N-1 cycle features cannot distinguish three patterns that are "), b("statistically identical cycle-by-cycle"), t(" but physically distinct:")]),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1800, 3960, 3600],
        rows: [
          headerRow(["Pattern", "Cycle Firing Sequence", "GAP vs BiGRU"], [1800, 3960, 3600]),
          dataRow(["A — Burst\n(7 consecutive)", "— — ARC ARC ARC ARC ARC ARC ARC — — — — — —", "GAP: sees 7 active, 43 silent. Same mean as Pattern B.\nBiGRU: hidden state accumulates arc evidence over 7 consecutive cycles \u2192 strong burst signal."], [1800, 3960, 3600], [C.white, C.white, C.white], [1,0,0]),
          dataRow(["B — Scattered\n(7 random cycles)", "ARC — — ARC — — — ARC — — ARC — — — — ARC —", "GAP: same mean as Pattern A. Identical feature vector.\nBiGRU: hidden state tracks intermittent firing with memory gaps \u2192 different embedding shape."], [1800, 3960, 3600], [C.white, C.white, C.white], [1,0,0]),
          dataRow(["C — Periodic Motor\n(false positive risk)", "ARC — ARC — ARC — ARC — ARC — ARC — ARC —", "GAP: 7 active cycles. Same mean as A and B. Cannot distinguish from arc.\nBiGRU: strict alternating periodicity encodes in hidden state \u2192 recognized as non-fault pattern."], [1800, 3960, 3600], [C.white, C.white, C.white], [1,0,0]),
        ]
      }),

      infoBox([
        para([b("Conclusion: ", C.blue), t("Patterns A, B, and C have identical global statistics (7/50 active cycles). GAP produces the same feature vector for all three. The BiGRU correctly differentiates them because its hidden state carries the temporal ordering, periodicity, and memory of past firings across the 50-cycle window.", C.text, 20)], { before: 0, after: 0 }),
      ], C.blue_light, C.blue),

      // ── Performance targets ───────────────────────────────────
      heading2("2.6  Performance Targets and Evaluation Protocol"),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2340, 2340, 2340, 2340],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: cellBorder(C.blue), shading: { fill: C.blue, type: ShadingType.CLEAR }, margins: { top: 120, bottom: 120, left: 160, right: 160 }, width: { size: 2340, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [
              para([new TextRun({ text: "\u226598%", bold: true, font: "Arial", size: 56, color: C.white })], { before: 0, after: 10, align: AlignmentType.CENTER }),
              para([new TextRun({ text: "Target Accuracy\nIEC 62606:2013 Eval", font: "Arial", size: 18, color: C.blue_light })], { before: 0, after: 0, align: AlignmentType.CENTER }),
            ]}),
            new TableCell({ borders: cellBorder(C.green), shading: { fill: C.green, type: ShadingType.CLEAR }, margins: { top: 120, bottom: 120, left: 160, right: 160 }, width: { size: 2340, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [
              para([new TextRun({ text: "7 / 50", bold: true, font: "Arial", size: 56, color: C.white })], { before: 0, after: 10, align: AlignmentType.CENTER }),
              para([new TextRun({ text: "Minimum Arc Cycles\nfor Reliable Detection", font: "Arial", size: 18, color: C.green_light })], { before: 0, after: 0, align: AlignmentType.CENTER }),
            ]}),
            new TableCell({ borders: cellBorder(C.amber), shading: { fill: C.amber, type: ShadingType.CLEAR }, margins: { top: 120, bottom: 120, left: 160, right: 160 }, width: { size: 2340, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [
              para([new TextRun({ text: "6", bold: true, font: "Arial", size: 56, color: C.white })], { before: 0, after: 10, align: AlignmentType.CENTER }),
              para([new TextRun({ text: "Simultaneous Loads\nin Masking Scenario", font: "Arial", size: 18, color: C.amber_light })], { before: 0, after: 0, align: AlignmentType.CENTER }),
            ]}),
            new TableCell({ borders: cellBorder(C.purple), shading: { fill: C.purple, type: ShadingType.CLEAR }, margins: { top: 120, bottom: 120, left: 160, right: 160 }, width: { size: 2340, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, children: [
              para([new TextRun({ text: "128", bold: true, font: "Arial", size: 56, color: C.white })], { before: 0, after: 10, align: AlignmentType.CENTER }),
              para([new TextRun({ text: "Embedding Dimensions\nInput to XGBoost", font: "Arial", size: 18, color: C.purple_light })], { before: 0, after: 0, align: AlignmentType.CENTER }),
            ]}),
          ]}),
        ]
      }),

      para([t(" ")], { before: 60, after: 0 }),
      rule(C.slate_border),
      para([t("ARC-FaultNet V2 Architecture Report  \u2014  Content brief for Canva slide creation", C.muted, 17)], { before: 40, after: 0, align: AlignmentType.CENTER }),

    ] // end children
  }] // end sections
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/mnt/user-data/outputs/ArcFaultNet_V2_Report.docx', buf);
  console.log('Done');
});

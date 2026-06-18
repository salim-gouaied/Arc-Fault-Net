import graphviz

def generate_diagram():
    dot = graphviz.Digraph('ArcFaultNetV2', format='png')
    dot.attr(rankdir='TD', size='12,12', fontname='Arial')
    dot.attr('node', fontname='Arial', shape='box', style='rounded,filled', fillcolor='white')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Inputs
    dot.node('In1D', 'Temporal Input (x_1d)\n(B, 4, M)\n[I, |ΔI|, TKEO, RMS]', shape='note', fillcolor='#e3f2fd')
    dot.node('In2D', 'Spectral Input (x_2d)\n(B, 1, F, T)\nLog-power STFT', shape='note', fillcolor='#e3f2fd')

    # Temporal Branch
    with dot.subgraph(name='cluster_temporal') as c:
        c.attr(label='Temporal Branch', style='dashed', bgcolor='#f5f5f5')
        c.node('T_Conv', 'Conv1d Stack\n(3x Conv1D + BN + GELU + MaxPool)', fillcolor='#ffffff')
        c.node('T_GAP', 'Global Average Pooling\nmean(dim=-1)', fillcolor='#ffffff')
        c.edge('T_Conv', 'T_GAP', label='(B, 128, D)')

    # Spectral Branch
    with dot.subgraph(name='cluster_spectral') as c:
        c.attr(label='Spectral Branch', style='dashed', bgcolor='#f5f5f5')
        c.node('S_Gate', 'FrequencyGate\nConv2D(3x1) + Sigmoid', fillcolor='#fce4ec')
        c.node('S_Conv', 'Conv2d Stack\n(Time Compression Pooling)', fillcolor='#ffffff')
        c.node('S_GAP', 'Global Average Pooling\nmean(dim=-1)', fillcolor='#ffffff')
        c.edge('S_Gate', 'S_Conv', label='Soft Attention Map')
        c.edge('S_Conv', 'S_GAP', label='(B, 128, D)')

    # Cross Attention Stage
    with dot.subgraph(name='cluster_cross_attn') as c:
        c.attr(label='Stage 4: RevisedCrossAttention', style='solid', bgcolor='#e8eaf6')
        c.node('ConcatJoint', 'Concatenate\njoint = [ f_t ; f_s ]', fillcolor='#ffffff')
        
        c.node('MLP_T', 'Temporal Channel Gate\nLinear(256→128) → ReLU → Linear → Sigmoid', fillcolor='#fce4ec')
        c.node('MLP_S', 'Spectral Channel Gate\nLinear(256→128) → ReLU → Linear → Sigmoid', fillcolor='#fce4ec')
        
        c.node('Mult_T', '✖️ Multiply', shape='circle', fillcolor='#ffffff')
        c.node('Mult_S', '✖️ Multiply', shape='circle', fillcolor='#ffffff')
        
        c.edge('ConcatJoint', 'MLP_T', label='joint (B, 256)')
        c.edge('ConcatJoint', 'MLP_S', label='joint (B, 256)')
        
        c.edge('MLP_T', 'Mult_T', label='α_temporal (B, 128)')
        c.edge('MLP_S', 'Mult_S', label='α_spectral (B, 128)')
        
        c.node('ConcatGated', 'Concatenate Gated Vectors\n[ f\'_t ; f\'_s ]', fillcolor='#ffffff')
        
        c.edge('Mult_T', 'ConcatGated', label='f\'_t')
        c.edge('Mult_S', 'ConcatGated', label='f\'_s')
        
        c.node('FusionMLP', 'Fusion Layer\nLinear(256→128) + GELU', fillcolor='#ffffff')
        c.edge('ConcatGated', 'FusionMLP', label='(B, 256)')

    # Output
    dot.node('EmbeddingOut', 'Final Embedded Vector\nshape: (B, 128)', shape='ellipse', fillcolor='#e8f5e9', style='filled')

    # Main Connections
    dot.edge('In1D', 'T_Conv')
    dot.edge('In2D', 'S_Gate')

    dot.edge('T_GAP', 'ConcatJoint', label='f_t (B, 128)')
    dot.edge('S_GAP', 'ConcatJoint', label='f_s (B, 128)')

    # Bypass connections for multiplication
    dot.edge('T_GAP', 'Mult_T', style='dotted', constraint='false')
    dot.edge('S_GAP', 'Mult_S', style='dotted', constraint='false')

    dot.edge('FusionMLP', 'EmbeddingOut')

    # Save the diagram
    output_path = '/home/manip/pfe_salim_gouaied/Arc-Fault-Net/arc_faultnet_v2_graphviz'
    dot.render(output_path, view=False, cleanup=True)
    print(f"Diagram saved to {output_path}.png")

if __name__ == '__main__':
    generate_diagram()

# Arc-FaultNet V2 Architecture & Embedded Vector Composition

The diagram below outlines exactly how the 128-dimensional embedding vector is generated in your `ArcFaultNetV2` model, breaking down the flow from the initial signal inputs through the `RevisedCrossAttention` gating mechanisms.

```mermaid
flowchart TD
    %% Inputs
    In1D["Temporal Input (x_1d)
    shape: (B, 4, M)
    [I, |ΔI|, TKEO, RMS]"]
    
    In2D["Spectral Input (x_2d)
    shape: (B, 1, F, T)
    Log-power STFT"]

    %% Branch Blocks
    subgraph TemporalBranch ["Temporal Branch"]
        T_Conv["Conv1d Stack + Pool
        (3x Conv1D + BN + GELU + MaxPool)"]
        T_GAP["Global Average Pooling
        mean(dim=-1)"]
    end

    subgraph SpectralBranch ["Spectral Branch"]
        S_Gate["FrequencyGate
        Conv2D(3x1) + Sigmoid"]
        S_Conv["Conv2d Stack + Asym. Pool
        (Time Compression)"]
        S_GAP["Global Average Pooling
        mean(dim=-1)"]
    end

    %% Branch Flow
    In1D --> T_Conv
    T_Conv --> |"shape: (B, 128, D)"| T_GAP
    T_GAP --> |"f_t shape: (B, 128)"| ConcatJoint

    In2D --> S_Gate
    S_Gate --> |Soft Attention| S_Conv
    S_Conv --> |"shape: (B, 128, D)"| S_GAP
    S_GAP --> |"f_s shape: (B, 128)"| ConcatJoint

    %% Revised Cross Attention
    subgraph CrossAttn ["Stage 4: RevisedCrossAttention"]
        direction TB
        ConcatJoint["Concatenate
        joint = [ f_t ; f_s ]"]
        
        subgraph TemporalGate ["Temporal Channel Gate"]
            MLP_T["MLP: Linear(256→128) 
            ReLU → Linear(128→128) → Sigmoid"]
        end
        
        subgraph SpectralGate ["Spectral Channel Gate"]
            MLP_S["MLP: Linear(256→128) 
            ReLU → Linear(128→128) → Sigmoid"]
        end
        
        Mult_T{"✖️ Multiply"}
        Mult_S{"✖️ Multiply"}
        
        ConcatJoint --> |"joint (B, 256)"| MLP_T
        ConcatJoint --> |"joint (B, 256)"| MLP_S
        
        MLP_T --> |"α_temporal (B, 128)"| Mult_T
        MLP_S --> |"α_spectral (B, 128)"| Mult_S
        
        ConcatGated["Concatenate Gated Vectors
        [ f'_t ; f'_s ]"]
        
        FusionMLP["Fusion Layer
        Linear(256→128) + GELU"]
    end
    
    %% Bypass connections to multiplier
    T_GAP -.-> |"f_t"| Mult_T
    S_GAP -.-> |"f_s"| Mult_S
    
    %% Down to fusion
    Mult_T --> |"f'_t = f_t ⊙ α_temporal"| ConcatGated
    Mult_S --> |"f'_s = f_s ⊙ α_spectral"| ConcatGated
    
    ConcatGated --> |"(B, 256)"| FusionMLP
    
    %% Output
    EmbeddingOut["Final Embedded Vector
    shape: (B, 128)
    (Passed to Classifier / XGBoost)"]
    
    FusionMLP --> EmbeddingOut

    %% Styling
    classDef default fill:#1a1a1a,stroke:#333,stroke-width:1px,color:#ddd;
    classDef input fill:#0f3443,stroke:#34e89e,stroke-width:2px,color:#fff;
    classDef output fill:#4a00e0,stroke:#8e2de2,stroke-width:2px,color:#fff;
    classDef attn fill:#4b134f,stroke:#c94b4b,stroke-width:2px,color:#fff;
    
    class In1D,In2D input;
    class EmbeddingOut output;
    class TemporalGate,SpectralGate,S_Gate attn;
```

### Step-by-Step Breakdown of the Embedded Vector Composition

**1. Branch Feature Extraction & Pooling**
*   **Temporal Branch:** The 4 derived 1D physical channels pass through a Conv1D stack. The output of shape `(Batch, 128, Time)` is squeezed down using a Global Average Pooling (mean over the time dimension) resulting in a single vector **$f_t$** of size `128` for each batch item.
*   **Spectral Branch:** The STFT input first goes through a **FrequencyGate** (a learnable 2D convolution that applies a soft sigmoid attention map to important frequency bands). The signal is then processed via a Conv2D stack with asymmetric pooling, and identically squeezed via GAP over the time axis into a vector **$f_s$** of size `128`.

**2. Context Concatenation (The "Joint" Vector)**
*   To enable the network to cross-reference information, $f_t$ and $f_s$ are directly joined together side-by-side: `joint = torch.cat([f_t, f_s], dim=-1)`. This results in a comprehensive **$256$-dimensional representation**.

**3. Cross-Conditioned Channel Gating (The Attention Mechanism)**
This is where the magic happens. Instead of applying channel attention to each branch in isolation, the channel importance for one branch is decided based on the **entire joint context**.
*   The `joint` vector is fed into two independent Multi-Layer Perceptrons (MLPs).
*   **Temporal Gate MLP:** Outputs `128` sigmoid weights ($\alpha_{temporal}$) representing the importance of the temporal features *given the entire system state*.
*   **Spectral Gate MLP:** Outputs `128` sigmoid weights ($\alpha_{spectral}$) representing the importance of the spectral features *given the entire system state*.

**4. Multiplication (Gating)**
*   The raw vector $f_t$ is multiplied channel-by-channel with $\alpha_{temporal}$ to get **$f'_t$**. Channels deemed unimportant by the MLP are squashed toward 0. 
*   The raw vector $f_s$ is multiplied channel-by-channel with $\alpha_{spectral}$ to get **$f'_s$**.

**5. Final Embedding Fusion**
*   The gated vectors are concatenated back together to shape `(Batch, 256)`.
*   A single Linear Layer (followed by a GELU activation) projects this down to the **final `128`-dimensional Embedded Vector**. 
*   This vector acts as the optimal, high-level summary of the arc fault and is passed either to the Phase-1 FC classification head, or extracted for use in Phase 2 (XGBoost/RandomForest).

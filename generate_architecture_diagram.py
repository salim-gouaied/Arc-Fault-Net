#!/usr/bin/env python3
"""Arc-FaultNet architecture diagram — v4 with visual hierarchy."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_TMP='#E07B54'; E_TMP='#A04020'
C_SPE='#2E9B80'; E_SPE='#1A6B55'
C_OUT='#4472C4'; E_OUT='#2F5597'
ANNOT='#444444'; ARR='#222222'; BG='white'

plt.rcParams['font.family']='DejaVu Sans'

def lc(h,a=.35):
    r,g,b=int(h[1:3],16)/255,int(h[3:5],16)/255,int(h[5:7],16)/255
    return '#{:02x}{:02x}{:02x}'.format(*[int((x+(1-x)*a)*255)for x in(r,g,b)])
def dc(h,a=.22):
    r,g,b=int(h[1:3],16)/255,int(h[3:5],16)/255,int(h[5:7],16)/255
    return '#{:02x}{:02x}{:02x}'.format(*[int(x*(1-a)*255)for x in(r,g,b)])

def fbox(ax,cx,cy,w,h,lbl,c,ec,fs=8,bold=False,z=3):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle="round,pad=0.04",facecolor=c,edgecolor=ec,lw=.9,zorder=z))
    ax.text(cx,cy,lbl,ha='center',va='center',fontsize=fs,color='white',
            weight='bold'if bold else'normal',zorder=z+1)

def attn_box(ax,cx,cy,w,h,lbl,c,ec,fs=8):
    """Octagonal attention box."""
    cl=0.10
    vx=[cx-w/2+cl,cx+w/2-cl,cx+w/2,cx+w/2,cx+w/2-cl,cx-w/2+cl,cx-w/2,cx-w/2]
    vy=[cy-h/2,cy-h/2,cy-h/2+cl,cy+h/2-cl,cy+h/2,cy+h/2,cy+h/2-cl,cy-h/2+cl]
    ax.add_patch(plt.Polygon(list(zip(vx,vy)),closed=True,
        facecolor=c,edgecolor=ec,lw=1.1,zorder=3))
    ax.text(cx,cy,lbl,ha='center',va='center',fontsize=fs,color='white',zorder=4)

def tensor3d(ax,cx,cy,w,h,c,ec,d=.12,z=3):
    """3-face isometric tensor (≥3D)."""
    fl,fr,fb,ft=cx-w/2,cx+w/2,cy-h/2,cy+h/2
    ax.add_patch(plt.Polygon([(fl,fb),(fr,fb),(fr,ft),(fl,ft)],
        closed=True,facecolor=c,edgecolor=ec,lw=.8,zorder=z))
    ax.add_patch(plt.Polygon([(fl,ft),(fr,ft),(fr+d,ft+d),(fl+d,ft+d)],
        closed=True,facecolor=lc(c,.40),edgecolor=ec,lw=.8,zorder=z))
    ax.add_patch(plt.Polygon([(fr,fb),(fr+d,fb+d),(fr+d,ft+d),(fr,ft)],
        closed=True,facecolor=dc(c,.22),edgecolor=ec,lw=.8,zorder=z))

def matrix2d(ax,cx,cy,w,h,c,ec,z=3):
    """Two overlapping flat rects = 2D matrix."""
    off=0.09
    ax.add_patch(FancyBboxPatch((cx-w/2+off,cy-h/2+off*.4),w,h,
        boxstyle="square,pad=0.0",facecolor=lc(c,.45),edgecolor=ec,lw=.7,
        alpha=.75,zorder=z-1))
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle="square,pad=0.0",facecolor=c,edgecolor=ec,lw=.9,zorder=z))

def vector1d(ax,cx,cy,w,h,c,ec,lbl,fs=7,z=3):
    """Single tall thin rect = 1D vector."""
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle="round,pad=0.02",facecolor=c,edgecolor=ec,lw=.9,zorder=z))
    ax.text(cx,cy,lbl,ha='center',va='center',fontsize=fs,color='white',zorder=z+1)

def stk(ax,cx,cy,w,h,lbl,c,ec,n=3,fs=7.5):
    off=0.06
    for i in range(n-1,0,-1):
        ax.add_patch(FancyBboxPatch((cx-w/2+i*off,cy-h/2+i*off),w,h,
            boxstyle="round,pad=0.04",facecolor=c,edgecolor=ec,lw=.8,alpha=.4,zorder=2))
    fbox(ax,cx,cy,w,h,lbl,c,ec,fs=fs)
    ax.text(cx+w/2+(n-1)*off+.08,cy+h/2+(n-1)*off+.02,f'×{n}',
            ha='left',va='bottom',fontsize=7.5,color=ANNOT)

def an(ax,cx,cy,t,fs=6.0,ha='center',va='top',style='italic'):
    ax.text(cx,cy,t,ha=ha,va=va,fontsize=fs,color=ANNOT,style=style)

def arrow(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=ARR,lw=.9,shrinkA=2,shrinkB=2),zorder=5)

def branch_rect(ax,x0,y0,x1,y1,lbl,c,va='top'):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle="round,pad=0.08",facecolor='none',edgecolor=c,
        lw=1.4,linestyle='--',zorder=1))
    ty=y1-.05 if va=='top' else y0+.05
    varg='top'if va=='top'else'bottom'
    ax.text(x0+.12,ty,lbl,ha='left',va=varg,fontsize=9,
            color=c,weight='bold',style='italic')

def enc_label(ax,cx,cy,lbl,c):
    ax.text(cx,cy,lbl,ha='center',va='center',fontsize=6.8,
            color=c,weight='bold',style='italic',
            bbox=dict(facecolor='white',edgecolor=c,lw=.7,pad=1.5,
                      boxstyle='round,pad=0.15'),zorder=5)

# ── Figure ──────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,6.0))
ax.set_xlim(0,11); ax.set_ylim(0,6.0)
ax.axis('off'); fig.patch.set_facecolor(BG)

# coordinate system: x∈[0,13], y∈[0,6]
# Temporal branch center y=4.5, Spectral y=2.0
# Right vertical stack x=11.2: xattn y=3.5, emb y=2.4, clf y=1.4, sig y=0.4

YT=4.30; YS=1.80
bw,bh=1.0,.52; aw,ah=.90,.52; sw,sh=1.05,.54
fw,fh=.65,.70; xw,xh=1.25,.62; cw,ch=1.05,.50

XI=0.65; XFE=1.95; XATN=3.20; XENC=5.00; XFM=6.60; XR=9.2

# ── INPUT ───────────────────────────────────────────────────────────────────
tensor3d(ax,XI,YT*.5+YS*.5,0.60,0.60,C_OUT,E_OUT,d=.10)
ax.text(XI,YT*.5+YS*.5,r'$I(t)$',ha='center',va='center',
        fontsize=10,color='white',zorder=5)
an(ax,XI,(YT+YS)/2-.48,'1×2048',fs=6)

arrow(ax,XI+.35,(YT+YS)/2+.14,XFE-bw/2,YT)
arrow(ax,XI+.35,(YT+YS)/2-.14,XFE-bw/2,YS)

# ── TEMPORAL BRANCH ─────────────────────────────────────────────────────────
branch_rect(ax,1.45,3.55,7.70,5.10,'Temporal Branch',C_TMP)

# Feature Engineering — tensor input (4×2048)
tensor3d(ax,XFE,YT,.90,.56,C_TMP,E_TMP,d=.11)
ax.text(XFE,YT,'Feature\nEng.',ha='center',va='center',fontsize=7.5,color='white',zorder=5)
an(ax,XFE,YT-.44,'4×2048',fs=6)

# DCA — attention (octagon)
attn_box(ax,XATN,YT,aw,ah,'DCA',C_TMP,E_TMP,fs=8)
an(ax,XATN,YT-.40,'Channel Attn.',fs=5.8)

# Temporal Encoder label
enc_label(ax,(XATN+XFM)/2+.3,YT+.52,'Temporal Encoder',C_TMP)

# Conv1d stack — tensor outputs
stk(ax,XENC,YT,sw,sh,'Conv1d+BN\n+GELU+DCA',C_TMP,E_TMP,n=3,fs=7)
an(ax,XENC,YT-.43,'32→64→128',fs=6)

# Branch output as MATRIX (128×D)
fbox(ax,XFM,YT,fw,fh,'Temporal\nFeatures',C_TMP,E_TMP,fs=7)
an(ax,XFM,YT-.48,'128×D',fs=6)

arrow(ax,XFE+.50,YT,XATN-aw/2,YT)
arrow(ax,XATN+aw/2,YT,XENC-sw/2,YT)
arrow(ax,XENC+sw/2+.17,YT,XFM-fw/2,YT)
arrow(ax,XFM+fw/2+.12,YT,XR-xw/2,4.05)

# ── SPECTRAL BRANCH ─────────────────────────────────────────────────────────
branch_rect(ax,1.45,.55,7.70,2.95,'Spectral Branch',C_SPE,va='bottom')

# STFT — tensor (1×F×T)
tensor3d(ax,XFE,YS,.90,.56,C_SPE,E_SPE,d=.11)
ax.text(XFE,YS,'STFT',ha='center',va='center',fontsize=8.5,color='white',zorder=5)
an(ax,XFE,YS-.44,'1×F×T',fs=6)

# Freq Gate — attention (octagon)
attn_box(ax,XATN,YS,aw,ah,'Freq.\nGate',C_SPE,E_SPE,fs=7.5)
an(ax,XATN,YS-.40,'Spectral Attn.',fs=5.8)

# Spectral Encoder label
enc_label(ax,(XATN+XFM)/2+.3,YS+.52,'Spectral Encoder',C_SPE)

# Conv2d stack — tensor outputs
stk(ax,XENC,YS,sw,sh,'Conv2d+BN\n+GELU',C_SPE,E_SPE,n=3,fs=7)
an(ax,XENC,YS-.43,'32→64→128',fs=6)

# Branch output as MATRIX (128×D)
fbox(ax,XFM,YS,fw,fh,'Spectral\nFeatures',C_SPE,E_SPE,fs=7)
an(ax,XFM,YS-.48,'128×D',fs=6)

arrow(ax,XFE+.50,YS,XATN-aw/2,YS)
arrow(ax,XATN+aw/2,YS,XENC-sw/2,YS)
arrow(ax,XENC+sw/2+.17,YS,XFM-fw/2,YS)
arrow(ax,XFM+fw/2+.12,YS,XR-xw/2,3.85)


# ── RIGHT VERTICAL STACK ────────────────────────────────────────────────────
# Sequential Cross-Attention (octagon — it IS an attention module)
cl2=.14; cxr,cyr,wr2,hr2=XR,3.95,xw,xh
vxa=[cxr-wr2/2+cl2,cxr+wr2/2-cl2,cxr+wr2/2,cxr+wr2/2,cxr+wr2/2-cl2,cxr-wr2/2+cl2,cxr-wr2/2,cxr-wr2/2]
vya=[cyr-hr2/2,cyr-hr2/2,cyr-hr2/2+cl2,cyr+hr2/2-cl2,cyr+hr2/2,cyr+hr2/2,cyr+hr2/2-cl2,cyr-hr2/2+cl2]
ax.add_patch(plt.Polygon(list(zip(vxa,vya)),closed=True,
    facecolor=C_OUT,edgecolor=E_OUT,lw=1.2,zorder=3))
ax.text(XR,3.95,'Sequential\nCross-Attention',ha='center',va='center',
        fontsize=8,color='white',weight='bold',zorder=4)
an(ax,XR,3.95-hr2/2-.08,'4-head Q/K/V',fs=6.2)

# Embedding — VECTOR (128,)
YEMB=2.98
vector1d(ax,XR,YEMB,.24,.72,C_OUT,E_OUT,'128',fs=7)
an(ax,XR+.20,YEMB,'embed.\nvector',fs=5.8,ha='left',va='center',style='normal')
arrow(ax,XR,3.95-hr2/2,XR,YEMB+.36)

# Classifier
YCLF=2.02
fbox(ax,XR,YCLF,cw,ch,'Classifier',C_OUT,E_OUT,fs=8.5)
an(ax,XR,YCLF-.34,'FC → GELU → FC',fs=6)
arrow(ax,XR,YEMB-.36,XR,YCLF+ch/2)

# σ output
YSIG=1.12
circ=plt.Circle((XR,YSIG),.22,facecolor=C_OUT,edgecolor=E_OUT,lw=.9,zorder=3)
ax.add_patch(circ)
ax.text(XR,YSIG,r'$\sigma$',ha='center',va='center',fontsize=11,color='white',zorder=4)
an(ax,XR,YSIG-.32,'P(arc) ∈ [0,1]',fs=6.2)
arrow(ax,XR,YCLF-ch/2,XR,YSIG+.22)

# ── Save ────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.1)
for ext,dpi in [('png',300),('pdf',None)]:
    kw=dict(bbox_inches='tight',facecolor='white',edgecolor='none')
    if dpi: kw['dpi']=dpi
    fig.savefig(f'arcfaultnet_architecture_paper.{ext}',**kw)
    print(f'Saved: arcfaultnet_architecture_paper.{ext}')

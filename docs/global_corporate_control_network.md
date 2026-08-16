# The Network of Global Corporate Control

**Paper**: Vitali, S., Glattfelder, J.B., & Battiston, S. (2011). *The network of global corporate control*. arXiv:1107.5728v2 [q-fin.GN].

## Key Findings

### Network Structure
- **Bow-tie topology** of the largest connected component (LCC): IN → SCC (core) → OUT, plus Tubes & Tendrils
- 600,508 nodes, 1,006,987 ownership ties from Orbis 2007 (43,060 TNCs + upstream/downstream)
- LCC contains 15,491 TNCs = 94.2% of total operating revenue

### The "Super-Entity" Core (SCC)
- **1,318 nodes**, 12,191 links — densely connected (avg 20 ties per node)
- **75% of core ownership stays within the core**
- **147 TNCs** control **38.5% of network control** (top 50 control 30.4%)
- **Financial institutions dominate**: 73% of top 50 control-holders are financial

### Control Computation
- **Network control** = direct control + indirect control via subsidiaries
- **Threshold model**: >50% ownership = full control; minority gets 0%
- Cycle correction: avoids overestimation in cross-shareholding structures

### Key Tables
| Section | TNCs | Operating Revenue % |
|---------|------|---------------------|
| LCC | 15,491 | 94.17% |
| IN | 282 | 2.18% |
| **SCC (core)** | **295** | **18.68%** |
| OUT | 6,488 | 59.85% |
| T&T | 8,426 | 13.46% |

## Relevance to Our Ownership Network

| Our Work | Paper Connection |
|----------|------------------|
| 13F-HR institutional holdings | Maps to "shareholder → participated company" edges |
| Look-through EV/EBITDA | Parallels "network control = economic value influenced" |
| Corporate control graph | Directly implements their network control methodology |
| Bow-tie detection | Can identify SCC in our network |

## Implementation Ideas

1. **Cycle detection** in our ownership network (cross-shareholdings)
2. **Threshold-based control** computation (currently just market-value weighted)
3. **Bow-tie decomposition** of our network (IN/SCC/OUT/T&T)
4. **Core-periphery analysis** to find "super-entity" in US institutional network
5. **Control flow metrics** beyond simple market value weighting

## Citation
```bibtex
@article{vitali2011network,
  title={The network of global corporate control},
  author={Vitali, Stefania and Glattfelder, James B and Battiston, Stefano},
  journal={PLoS ONE},
  volume={6},
  number={10},
  pages={e25995},
  year={2011},
  publisher={Public Library of Science}
}
```
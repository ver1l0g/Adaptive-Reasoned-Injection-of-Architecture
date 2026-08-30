# ARIA Freeze Card (M6.5/M6.7) — compiled 2026-08-30

Binary lineage: aria12 (aria10 + M7.5/M7.6 live + EMBED UAF fix + M1.5
versioning) for all suites; aria11 for standard/korns/temporal (identical
engine semantics to aria12 on those paths); aria16 spot-checks noted in
place. Post-freeze development binaries (aria13-18: evidence path,
one-hot) are NOT in this card — they are v2.
Development binaries & deltas: see ROADMAP M6.7 for the freeze-binary
decision record.

## suite-standard (17 tasks, aria11)
13/16 pass + d9 noise-floor

| d1_step | 0.992956 |
| d2_mod3 | 0.758130 | **OPEN**
| d3_two_sines | 0.553664 | **OPEN**
| d4_irrelevant12 | 0.999244 |
| d5_extrap_cliff | 0.999863 |
| d6_seq_parity | 1.000000 |
| d7_multiout | 0.999396 |
| d8_compose_absprod | 0.999576 |
| d9_noise | 0.806034 | (noise floor)
| t21_three_region | 0.993627 |
| t22_windowed_sine | 0.964378 | **OPEN**
| t23_quadratic | 0.997837 |
| t24_quadrant_xor | 0.995615 |
| t31_three_way_product | 0.994887 |
| t32_sine_of_product | 0.998731 |
| t33_absolute_value | 0.998408 |
| t34_xor3_parity | 0.997394 |

## suite-korns (9 tasks, aria11)
6/9 > 0.99

| F1 | 0.999837 |
| F2 | 0.999842 |
| F3 | 0.999884 |
| F4 | 0.982698 | **OPEN**
| F5 | 0.998260 |
| F6 | 0.998598 |
| F7 | 0.998713 |
| F8 | 0.859121 | **OPEN**
| F9 | 0.988300 | **OPEN**

## suite-temporal (aria11)
```
lorenz3      val=0.054141     train=0.0527835    commits=COMPOUND_DIVIDE_PRODUCTx3,MULTI_LAYER_STACKx1,MULTIPLY_INJECTIONx1,DIVIDE_INJECTIONx1 (40s)
lorenz       val=0.027853     train=0.0271987    commits=CONTEXT_WIREx1,MULTI_LAYER_STACKx1,MUX_INJECTIONx1,RECURRENT_MULTI_TAPx1 (198s)
narma10_lag  val=0.004896     train=0.00684376   commits=RECURRENT_MULTI_TAPx1,COMPOUND_SIN_PRODUCTx1,MULTIPLY_INJECTIONx1 (73s)
narma10      val=0.009286     train=0.0118832    commits=DELAY_LINEx2,RECURRENT_MULTI_TAPx4,DEEP_INSERTIONx2 (124s)
```

## suite-limits (aria12; solo reruns where the 5-way runs timed out)
```
highdim15   no-eval  commits=MULTI_LAYER_STACKx5  (30min)
highdim20   no-eval  commits=MULTI_LAYER_STACKx3,MULTIPLY_INJECTIONx1,CONTEXT_WIREx1  (30min)
narma30     out1=0.0050[FAIL]  commits=DELAY_LINEx2,RECURRENT_MULTI_TAPx3,DEEP_INSERTIONx2,MUX_INJECTIONx1  (5min)
hetero3     out1=1.0000[PASS] out2=0.9965[PASS] out3=0.9899[FAIL]  commits=NEURON_TANH_INJECTIONx12,SIN_INJECTIONx4,MULTIPLY_INJECTIONx13,IFELSE_BOUNDARY_SPLITx2,DEEP_INSERTIONx4,CONTEXT_WIREx8,MUX_INJECTIONx1,MULTI_LAYER_STACKx1  (8min)
stripes20   out1=0.1043[FAIL]  commits=IFELSE_PRESERVEx4,NEURON_TANH_INJECTIONx4,BOOLEAN_COMPOSEx1,MUX_INJECTIONx1,SIN_INJECTIONx4,IFELSE_BOUNDARY_SPLITx1,CONTEXT_WIREx4  (3min)
count8      out1=0.9999[PASS]  commits=NEURON_TANH_INJECTIONx4,MULTIPLY_INJECTIONx1,IFELSE_BOUNDARY_SPLITx1,CONTEXT_WIREx1  (1min)
firstpos8   out1=0.9858[FAIL]  commits=NEURON_TANH_INJECTIONx1,MULTIPLY_INJECTIONx2  (1min)

| highdim20 solo rerun | 0.995278 PASS |
| highdim15 solo rerun | 0.905612 (aria10 freeze: 0.9982 —
|                      |  build/trajectory-sensitive, see M6.8) |
```

## suite-feynman multiseed (aria12)
aria12, 50 epochs, seeds 2-5 (aria10 archive: results/aria10_multiseed/)
mean-of-means 0.9991 | solved 24/25

| I.10.7 | 0.9997 ± 0.0001 (n=4) |
| I.12.5 | 0.9999 ± 0.0002 (n=4) |
| I.13.12 | 0.9999 ± 0.0001 (n=4) |
| I.14.3 | 0.9997 ± 0.0000 (n=4) |
| I.15.10 | 0.9998 ± 0.0001 (n=4) |
| I.15.3 | 0.9997 ± 0.0001 (n=4) |
| I.18.14 | 0.9999 ± 0.0000 (n=4) |
| I.18.4 | 0.9999 ± 0.0000 (n=4) |
| I.24.6 | 0.9999 ± 0.0000 (n=4) |
| I.25.9 | 0.9999 ± 0.0001 (n=4) |
| I.29.16 | 0.9993 ± 0.0001 (n=4) |
| I.32.8 | 0.9829 ± 0.0017 (n=4) | **OPEN**
| I.34.6 | 0.9991 ± 0.0000 (n=4) |
| I.43.27 | 0.9984 ± 0.0005 (n=4) |
| I.47.23 | 0.9999 ± 0.0000 (n=4) |
| I.48.20 | 0.9999 ± 0.0000 (n=3) |
| I.6.2 | 1.0000 ± 0.0000 (n=4) |
| I.6.2b | 0.9998 ± 0.0002 (n=4) |
| I.7.9 | 1.0000 ± 0.0000 (n=3) |
| I.8.4 | 0.9999 ± 0.0001 (n=4) |
| I.9.5 | 0.9995 ± 0.0004 (n=4) |
| II.11.27 | 0.9998 ± 0.0002 (n=4) |
| II.34.29a | 0.9998 ± 0.0001 (n=4) |
| III.10.19 | 0.9999 ± 0.0000 (n=4) |
| III.4.33 | 1.0000 ± 0.0000 (n=4) |

## suite-language ladder (aria12, EMBED trunk)
| w1 | 4.429599 bits/char |
| w8 | 4.227126 bits/char |
| w16 | 4.171213 bits/char |
| w32 | 4.462500 bits/char |

Monotone through w16 (4.43 -> 4.17); stall at w32 confirmed
(solo rerun 4.4625; historical 4.231). M2.4: attention GO.

/*
 * AHB Bus SPICE Model
 * ===================
 *
 * Models the Advanced High-performance Bus (AHB) in CC2650.
 * Includes wire resistance, capacitance, and timing.
 */

* Title: CC2650 AHB Bus
* Bus width: 32 bits
* Frequency: 48 MHz
* Technology: TSMC 65nm

* Supply
VDD VDD 0 DC 3.3V
VGND GND 0 DC 0V

* Wire models (distributed RC)
* Each wire segment
.subckt wire_segment in out length=100u width=1u
R1 in n1 R='1.0u/length*0.1'
R2 n1 out R='1.0u/length*0.1'
C1 in GND C='2fF/length*width'
C2 n1 GND C='1fF/length*width'
C3 out GND C='2fF/length*width'
.ends wire_segment

* Address bus (32 bits, 0.5mm length)
XADDR0 A0 A0_OUT VDD GND wire_segment length=500u width=1u
XADDR1 A1 A1_OUT VDD GND wire_segment length=500u width=1u
XADDR2 A2 A2_OUT VDD GND wire_segment length=500u width=1u
XADDR3 A3 A3_OUT VDD GND wire_segment length=500u width=1u
XADDR4 A4 A4_OUT VDD GND wire_segment length=500u width=1u
XADDR5 A5 A5_OUT VDD GND wire_segment length=500u width=1u
XADDR6 A6 A6_OUT VDD GND wire_segment length=500u width=1u
XADDR7 A7 A7_OUT VDD GND wire_segment length=500u width=1u
XADDR8 A8 A8_OUT VDD GND wire_segment length=500u width=1u
XADDR9 A9 A9_OUT VDD GND wire_segment length=500u width=1u
XADDR10 A10 A10_OUT VDD GND wire_segment length=500u width=1u
XADDR11 A11 A11_OUT VDD GND wire_segment length=500u width=1u
XADDR12 A12 A12_OUT VDD GND wire_segment length=500u width=1u
XADDR13 A13 A13_OUT VDD GND wire_segment length=500u width=1u
XADDR14 A14 A14_OUT VDD GND wire_segment length=500u width=1u
XADDR15 A15 A15_OUT VDD GND wire_segment length=500u width=1u
XADDR16 A16 A16_OUT VDD GND wire_segment length=500u width=1u
XADDR17 A17 A17_OUT VDD GND wire_segment length=500u width=1u
XADDR18 A18 A18_OUT VDD GND wire_segment length=500u width=1u
XADDR19 A19 A19_OUT VDD GND wire_segment length=500u width=1u
XADDR20 A20 A20_OUT VDD GND wire_segment length=500u width=1u
XADDR21 A21 A21_OUT VDD GND wire_segment length=500u width=1u
XADDR22 A22 A22_OUT VDD GND wire_segment length=500u width=1u
XADDR23 A23 A23_OUT VDD GND wire_segment length=500u width=1u
XADDR24 A24 A24_OUT VDD GND wire_segment length=500u width=1u
XADDR25 A25 A25_OUT VDD GND wire_segment length=500u width=1u
XADDR26 A26 A26_OUT VDD GND wire_segment length=500u width=1u
XADDR27 A27 A27_OUT VDD GND wire_segment length=500u width=1u
XADDR28 A28 A28_OUT VDD GND wire_segment length=500u width=1u
XADDR29 A29 A29_OUT VDD GND wire_segment length=500u width=1u
XADDR30 A30 A30_OUT VDD GND wire_segment length=500u width=1u
XADDR31 A31 A31_OUT VDD GND wire_segment length=500u width=1u

* Data bus (32 bits)
XDATA0 D0 D0_OUT VDD GND wire_segment length=500u width=2u
XDATA1 D1 D1_OUT VDD GND wire_segment length=500u width=2u
XDATA2 D2 D2_OUT VDD GND wire_segment length=500u width=2u
XDATA3 D3 D3_OUT VDD GND wire_segment length=500u width=2u
XDATA4 D4 D4_OUT VDD GND wire_segment length=500u width=2u
XDATA5 D5 D5_OUT VDD GND wire_segment length=500u width=2u
XDATA6 D6 D6_OUT VDD GND wire_segment length=500u width=2u
XDATA7 D7 D7_OUT VDD GND wire_segment length=500u width=2u
XDATA8 D8 D8_OUT VDD GND wire_segment length=500u width=2u
XDATA9 D9 D9_OUT VDD GND wire_segment length=500u width=2u
XDATA10 D10 D10_OUT VDD GND wire_segment length=500u width=2u
XDATA11 D11 D11_OUT VDD GND wire_segment length=500u width=2u
XDATA12 D12 D12_OUT VDD GND wire_segment length=500u width=2u
XDATA13 D13 D13_OUT VDD GND wire_segment length=500u width=2u
XDATA14 D14 D14_OUT VDD GND wire_segment length=500u width=2u
XDATA15 D15 D15_OUT VDD GND wire_segment length=500u width=2u
XDATA16 D16 D16_OUT VDD GND wire_segment length=500u width=2u
XDATA17 D17 D17_OUT VDD GND wire_segment length=500u width=2u
XDATA18 D18 D18_OUT VDD GND wire_segment length=500u width=2u
XDATA19 D19 D19_OUT VDD GND wire_segment length=500u width=2u
XDATA20 D20 D20_OUT VDD GND wire_segment length=500u width=2u
XDATA21 D21 D21_OUT VDD GND wire_segment length=500u width=2u
XDATA22 D22 D22_OUT VDD GND wire_segment length=500u width=2u
XDATA23 D23 D23_OUT VDD GND wire_segment length=500u width=2u
XDATA24 D24 D24_OUT VDD GND wire_segment length=500u width=2u
XDATA25 D25 D25_OUT VDD GND wire_segment length=500u width=2u
XDATA26 D26 D26_OUT VDD GND wire_segment length=500u width=2u
XDATA27 D27 D27_OUT VDD GND wire_segment length=500u width=2u
XDATA28 D28 D28_OUT VDD GND wire_segment length=500u width=2u
XDATA29 D29 D29_OUT VDD GND wire_segment length=500u width=2u
XDATA30 D30 D30_OUT VDD GND wire_segment length=500u width=2u
XDATA31 D31 D31_OUT VDD GND wire_segment length=500u width=2u

* Clock tree
.subckt clock_tree clk_in clk_out level=5
R1 clk_in n1 10
C1 n1 GND 50f
Xbuf n1 clk_out VDD GND buffer
.ends clock_tree

XCLKBUF HCLK HCLK_OUT VDD GND clock_tree level=5

* Control signals
XHWRITE HWRITE HWRITE_OUT VDD GND wire_segment length=300u width=1u
XHSIZE HSIZE0 HSIZE0_OUT VDD GND wire_segment length=300u width=1u
XHSIZE1 HSIZE1 HSIZE1_OUT VDD GND wire_segment length=300u width=1u
XHBURST HBURST0 HBURST0_OUT VDD GND wire_segment length=300u width=1u

* Bus capacitance (total C_bus = 12.3pF as measured)
C_BUS VDD GND 12.3p

* Input drivers
VADDR A0 0 PULSE(0 3.3 0 100ps 100ps 5ns 20ns)
VCLK HCLK 0 PULSE(0 3.3 0 500ps 500ps 10ns 20ns)

* Analysis
.control
tran 100ps 200ns
plot V(A0) V(D0_OUT) V(HCLK_OUT)
.endc

* Models
.lib /path/to/tsmc65nm.lib tt
.end

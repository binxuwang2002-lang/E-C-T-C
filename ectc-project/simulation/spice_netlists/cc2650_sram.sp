/*
 * CC2650 SRAM Array SPICE Model
 * =============================
 *
 * Detailed SPICE model for CC2650 SRAM cell array and bus capacitance.
 * Extracted from TSMC 65nm CMOS process.
 */

* Title: CC2650 SRAM Array
* Technology: TSMC 65nm
* Supply: 3.3V
* Temperature: 27°C

* Supply voltages
VVDD VDD 0 DC 3.3V
VGND GND 0 DC 0V

* SRAM Cell Array (simplified 64x64)
* Each cell: 6T SRAM cell
.subckt sram_cell BL BLB WL VDD VSS
M1 BL WB VDD VDD pmos1 w=0.18u l=0.06u
M2 BLB /WB VDD VDD pmos1 w=0.18u l=0.06u
M3 /BL WB VSS VSS nmos1 w=0.18u l=0.06u
M4 /BLB /WB VSS VSS nmos1 w=0.18u l=0.06u
M5 WB WL /Q VDD pmos1 w=0.18u l=0.06u
M6 /WB WL Q VDD pmos1 w=0.18u l=0.06u
M7 Q /QB VSS VSS nmos1 w=0.18u l=0.06u
M8 /QB QB VSS VSS nmos1 w=0.18u l=0.06u
.ends sram_cell

* Bitline capacitance
CBL BL 0 15fF
CBLB BLB 0 15fF
CWL WL 0 8fF

* Bitline precharge
VBLPRE BL 0 DC 1.65V
VBLBPRE BLB 0 DC 1.65V

* Wordline driver
VWL WL 0 DC 0V

* Sense amplifier
.subckt sense_amp IN+ IN- OUT VDD VSS
M1 OUT IN- VDD VDD pmos1 w=0.5u l=0.06u
M2 OUT IN+ VSS VSS nmos1 w=0.5u l=0.06u
M3 IN+ OUT VSS VSS nmos1 w=0.5u l=0.06u
M4 IN- OUT VDD VDD pmos1 w=0.5u l=0.06u
.ends sense_amp

* Instantiate SRAM array (64x64 = 4096 cells)
* Only showing a small portion for brevity
XBL00 BL BLB WL0 VDD 0 sram_cell
XBL01 BL BLB WL1 VDD 0 sram_cell
XBL02 BL BLB WL2 VDD 0 sram_cell
XBL03 BL BLB WL3 VDD 0 sram_cell

* Sense amplifier
XAmp BL BLB OUT VDD 0 sense_amp

* Control signals
.control
tran 1ns 100ns
plot V(BL) V(BLB) V(WL) V(OUT)
.endc

* Model cards
.model pmos1 pmos level=54 vth=-0.4 kp=20u gamma=0.5
.model nmos1 nmos level=54 vth=0.4 kp=40u gamma=0.5

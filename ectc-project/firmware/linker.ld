/*
** ###################################################################
**     Linker file for the GNU C Compiler
**
**     CC2650F128RGZ
**     128K bytes (alias Flash  1) total on-chip Flash memory
**     8K bytes (alias Ram0_1_0) total on-chip RAM
**
**     Based on Texas Instruments linker files
**
**     Version: 1.0.0
**     Build:   b180801
**
** ###################################################################
*/

/*
 *  THIS LINKER FILE WAS MODIFIED, BUT MAINTAINED FOR ECTC PROJECT
 *
 *  Memory allocation for ECTC:
 *  - Flash:    TinyLSTM weights (4KB), Pedersen commitments, application code
 *  - SRAM:     Runtime buffers, LSTM hidden states, trace buffer (200 bytes)
 *  - FRAM:     Configuration data (not available in CC2650, would use external)
 */

/* Entry Point */
ENTRY(ResetISR)

/* Specify the memory areas */
MEMORY
{
    /* Define each memory region */
    /* Define a symbol name to be used when checking for memory overflows */
    FLASH (rx) : ORIGIN = 0x0, LENGTH = 128K /* 128K bytes (alias Flash) */
    /*
     * Memory for FRAM would be added here if using external FRAM:
     * FRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 8K
     */

    /* SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 8K */
    /* Use smaller SRAM sections for better control */
    SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 28K      /* Main SRAM (28KB) */
    SRAM2 (rwx) : ORIGIN = 0x20007000, LENGTH = 8K      /* Additional SRAM (8KB) */
    /* Keep 4KB for system (timers, radio, etc.) */

    /* Define a symbol that marks the last valid RAM address */
    _estack = 0x2000FFFF;    /* 64KB SRAM total end */

    /*
     * Note: CC2650 has 40KB total SRAM
     * This linker script allocates:
     * - 28KB for application (SRAM)
     * - 8KB for additional buffers (SRAM2)
     * - 4KB reserved for system
     */
}

/* Define a symbol to mark the last valid flash address */
_flash_end = 0x00020000;  /* 128KB flash end */

/* Define a symbol for the boundary between SRAM and SRAM2 */
_sram_end = 0x20007FFF;   /* 32KB SRAM end */

/* Specify the default entry point */
SECTIONS
{
    .ResetISR : ALIGN(4)
    {
        KEEP(*(.isr_vector))
        /*
         * For CC2650, the interrupt vector table is at the start of flash
         * This section should contain the ResetISR and vector table
         */
    } >FLASH

    .text : ALIGN(4)
    {
       FILL(0xff)
       __vectors_start__ = ABSOLUTE(.) ;
       KEEP(*(.vectors))
       __vectors_end__ = ABSOLUTE(.) ;
       *(.text*)
       *(.rodata .rodata* .constdata .constdata* .conststring .conststring*)
       . = ALIGN(4);
    } >SRAM AT>FLASH /* Load in flash, run in SRAM for speed */

    /*
     * For ECTC: store TinyLSTM weights in flash
     * This allows us to keep weights persistent while minimizing SRAM usage
     */
    .text.rodata : ALIGN(4)
    {
       FILL(0xff)
       *(.rodata .rodata* .constdata .constdata*)
       . = ALIGN(4);
    } >FLASH

    /* Place TinyLSTM weights in dedicated section */
    .tinylstm_weights : ALIGN(4)
    {
       *(.text.lstm_weights)
       *(.text.tiny_lstm*)
       _tinylstm_weights_start = .;
       KEEP(*(.fram_data))  /* If using external FRAM */
       _tinylstm_weights_end = .;
    } >FLASH

    /*
     * Persistent data (would be in FRAM if available)
     * For now, use flash to simulate FRAM persistence
     */
    .persistent : ALIGN(4)
    {
       _persistent_start = .;
       *(.persistent*)
       _persistent_end = .;
    } >FLASH

    /*
     * C++ constructor and destructor tables
     */
    .init_array : ALIGN(4)
    {
       FILL(0xff)
       _init_array_start = .;
       KEEP (*(SORT(.init_array.*)))
       KEEP (*(.init_array))
       _init_array_end = .;
    } >SRAM AT>FLASH

    .fini_array : ALIGN(4)
    {
       FILL(0xff)
       _fini_array_start = .;
       KEEP (*(SORT(.fini_array.*)))
       KEEP (*(.fini_array))
       _fini_array_end = .;
    } >SRAM AT>FLASH

    /*
     * Uninitialized data (zero-initialized)
     * These symbols are marked with KEEP so they are not garbage collected
     */
    .uninit_RESERVED : ALIGN(4)
    {
       _start_uninit_RESERVED = .;
       KEEP(*(.bss.$RESERVED*))
       . = ALIGN(4) ;
       _end_uninit_RESERVED = .;
    } >SRAM

    /*
     * For ECTC: Reserve space for trace buffer
     * 200-byte trace buffer as specified in requirements
     */
    .trace_section : ALIGN(4)
    {
       _trace_start = .;
       KEEP(*(.retain))
       _trace_end = .;
    } >SRAM

    /*
     * C++ exception handling tables
     */
    .ARM.extab : ALIGN(4)
    {
       FILL(0xff)
       *(.ARM.extab* .gnu.linkonce.armextab.*)
       . = ALIGN(4);
    } >SRAM AT>FLASH

    __exidx_start = .;
    .ARM.exidx : ALIGN(4)
    {
       FILL(0xff)
       *(.ARM.exidx* .gnu.linkonce.armexidx.*)
       . = ALIGN(4);
    } >SRAM AT>FLASH
    __exidx_end = .;

    /*
     * Data section for SRAM2 (additional 8KB buffer space)
     */
    .sram2_data : ALIGN(4)
    {
       _sram2_start = .;
       *(.data.sram2*)
       *(.bss.sram2*)
       . = ALIGN(4);
       _sram2_end = .;
    } >SRAM2 AT>SRAM2

    /*
     * Standard C Runtime Data
     */
    .data : ALIGN(4)
    {
       FILL(0xff)
       _data_start = . ;
       *(.ramfunc*)
       *(.data*)
       . = ALIGN(4) ;
       _data_end = .;
    } >SRAM AT>FLASH

    /*
     * Uninitialized data
     */
    .uninit_RESERVED : ALIGN(4)
    {
       _start_uninit_RESERVED = .;
       KEEP(*(.bss.$RESERVED*))
       . = ALIGN(4) ;
       _end_uninit_RESERVED = .;
    } >SRAM

    /*
     * BSS Section for RAM
     */
    .bss : ALIGN(4)
    {
       _bss_start = .;
       *(.bss*)
       *(COMMON)
       . = ALIGN(4) ;
       _bss_end = .;
    } >SRAM

    /*
     * BSS Section for SRAM2
     */
    .bss_sram2 : ALIGN(4)
    {
       _bss_sram2_start = .;
       *(.bss.sram2*)
       *(COMMON)
       . = ALIGN(4) ;
       _bss_sram2_end = .;
    } >SRAM2

    /*
     * DEFAULT NOINIT SECTION
     */
    .noinit_RESERVED : ALIGN(4)
    {
       _noinit = .;
       *(.noinit*)
        . = ALIGN(4) ;
       _end_noinit = .;
    } >SRAM

    /*
     * Define heap and stack regions
     */
    .heap : ALIGN(8)
    {
       _HeapBase = .;
       . = . + 0x2000;  /* 8KB heap */
       . = ALIGN(8);
       _HeapLimit = .;
    } >SRAM

    .stack : ALIGN(8)
    {
       _StackLimit = _estack - 0x1000;  /* Reserve 4KB for system stack */
       _StackTop = _estack;
    } >SRAM

    /*
     * ECTC: Reserve space for energy monitoring
     * Add energy history buffer (40 entries)
     */
    .energy_buffer : ALIGN(4)
    {
       _energy_buffer_start = .;
       *(.energy_buffer*)
       _energy_buffer_end = .;
       . = . + 160;  /* 40 * 4 bytes */
    } >SRAM
}

/* Provide basic symbols giving location and size of main text
 * block, including initial values of RW data sections. */
_image_start = LOADADDR(.text);
_image_end = LOADADDR(.data) + SIZEOF(.data);
_image_size = _image_end - _image_start;

/* End of Linker Script */

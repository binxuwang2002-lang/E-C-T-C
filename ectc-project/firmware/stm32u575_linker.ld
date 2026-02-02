/*
** ###################################################################
**     Linker file for STM32U575 (Cortex-M33)
**
**     Memory Configuration based on ECTC Static Memory Allocation Strategy
**     (Reference: ECTC-19.pdf, Table I, Page 7)
**
**     Hardware: STM32U575 with 40KB SRAM limit
**
**     Memory Layout:
**     - TinyLSTM Weights: 8KB at 0x20000000
**     - Shapley Cache: 4KB at 0x20002000
**     - BSS/Buffers: 20KB
**     - Retention RAM: 8KB at 0x20006400
**
**     Version: 1.0.0
**     Build: 2024
**
** ###################################################################
*/

/* Entry Point */
ENTRY(ResetISR)

/*
 * Memory Map for STM32U575
 * ------------------------
 * - Flash:  2MB (0x08000000 - 0x081FFFFF)
 * - SRAM:   40KB (0x20000000 - 0x20009FFF)
 */

/* Specify the memory areas */
MEMORY
{
    /* Flash memory - 2MB for code and constants */
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 2048K

    /* SRAM - 40KB unified memory region */
    SRAM_CORE (rwx) : ORIGIN = 0x20000000, LENGTH = 40K

    /* Define symbols for memory bounds */
    _estack = 0x20009FFF;    /* End of SRAM (40KB total) */
    _Min_Heap_Size = 0x1000; /* 4KB Heap */
    _Min_Stack_Size = 0x3000; /* 12KB Stack */
}

/* Define a symbol for the last valid flash address */
_flash_end = 0x08000000 + 2048K;

/* Specify the default entry point */
SECTIONS
{
    /* Interrupt Vector Table */
    .isr_vector : ALIGN(4)
    {
        _isr_vector_start = .;
        KEEP(*(.isr_vector))
        _isr_vector_end = .;
    } >FLASH

    /* Code and read-only data */
    .text : ALIGN(4)
    {
       FILL(0xff)
       *(.text*)
       *(.rodata .rodata* .constdata .constdata*)
       . = ALIGN(4);
    } >FLASH

    /*
     * TinyLSTM Weights Section (8KB)
     * (Per ECTC Table I: Static Memory Allocation Strategy)
     * Located at 0x20000000 (start of SRAM)
     */
    .tinylstm_weights 0x20000000 : ALIGN(4)
    {
       _tinylstm_weights_start = .;
       KEEP(*(.tinylstm_weights))
       _tinylstm_weights_end = .;
       _tinylstm_weights_size = _tinylstm_weights_end - _tinylstm_weights_start;
    } >SRAM_CORE

    /*
     * Shapley Cache Section (4KB)
     * (Per ECTC Table I: Static Memory Allocation Strategy)
     * Located at 0x20002000
     * Allows O(1) lookup during gateway failure
     */
    .shapley_cache 0x20002000 : ALIGN(4)
    {
       _shapley_cache_start = .;
       KEEP(*(.shapley_cache))
       _shapley_cache_end = .;
       _shapley_cache_size = _shapley_cache_end - _shapley_cache_start;
    } >SRAM_CORE

    /* Initialized data (loaded to Flash, copied to RAM at startup) */
    .data : ALIGN(4)
    {
       _data_start = . ;
       *(.ramfunc*)
       *(.data*)
       . = ALIGN(4) ;
       _data_end = .;
    } >SRAM_CORE AT>FLASH

    /*
     * Uninitialized data (BSS)
     * Network Stack & App Buffers - 20KB
     */
    .bss : ALIGN(4)
    {
       _bss_start = .;
       *(.bss*)
       *(COMMON)
       . = ALIGN(4) ;
       _bss_end = .;
    } >SRAM_CORE

    /*
     * Retention RAM Section (8KB at end of SRAM)
     * Checkpoints strictly mapped here (Per ECTC Table I)
     * Located at 0x20006400
     */
    .retention_ram 0x20006400 : ALIGN(4)
    {
       _retention_ram_start = .;
       KEEP(*(.retention_ram))
       KEEP(*(.retention_checkpoint))
       KEEP(*(.noinit))
       _retention_ram_end = .;
       _retention_ram_size = _retention_ram_end - _retention_ram_start;
    } >SRAM_CORE

    /* Heap - placed at end of SRAM */
    .heap : ALIGN(8)
    {
       _HeapBase = .;
       . = . + _Min_Heap_Size;
       . = ALIGN(8);
       _HeapLimit = .;
    } >SRAM_CORE

    /* Stack - placed at end of SRAM */
    .stack : ALIGN(8)
    {
       _StackLimit = _estack - _Min_Stack_Size;
       _StackTop = _estack;
    } >SRAM_CORE

    /* C++ exception handling tables */
    .ARM.extab : ALIGN(4)
    {
       FILL(0xff)
       *(.ARM.extab* .gnu.linkonce.armextab.*)
       . = ALIGN(4);
    } >SRAM_CORE AT>FLASH

    __exidx_start = .;
    .ARM.exidx : ALIGN(4)
    {
       FILL(0xff)
       *(.ARM.exidx* .gnu.linkonce.armexidx.*)
       . = ALIGN(4);
    } >SRAM_CORE AT>FLASH
    __exidx_end = .;

    /* Provide basic symbols giving location and size of main text
     * block, including initial values of RW data sections. */
    _image_start = LOADADDR(.text);
    _image_end = LOADADDR(.data) + SIZEOF(.data);
    _image_size = _image_end - _image_start;
}

/*
 * Memory Usage Verification
 * Ensures static memory allocation stays within ECTC Table I limits
 */
PROVIDE( _tinylstm_used = _tinylstm_weights_size );
PROVIDE( _shapley_cache_used = _shapley_cache_size );
PROVIDE( _retention_ram_used = _retention_ram_size );

/* Static assertions for memory limits */
ASSERT( _tinylstm_weights_size <= 8K, "TinyLSTM weights exceed 8KB limit" );
ASSERT( _shapley_cache_size <= 4K, "Shapley cache exceeds 4KB limit" );
ASSERT( _retention_ram_size <= 8K, "Retention RAM exceeds 8KB limit" );

/* End of Linker Script */

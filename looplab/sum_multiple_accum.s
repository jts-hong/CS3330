// This is assembly is roughly equivalent to the following C code:
// unsigned short sum_C(long size, unsigned short * a) {
//    unsigned short sum = 0;
//    for (int i = 0; i < size; ++i) {
//        sum += a[i];
//    }
//    return sum;
//}

// This implementation follows the Linux x86-64 calling convention:
//    %rdi contains the size
//    %rsi contains the pointer a
// and
//    %ax needs to contain the result when the function returns
// in addition, this code uses
//    %rcx to store i

// the '.global' directive indicates to the assembler to make this symbol 
// available to other files.
.global sum_multiple_accum
sum_multiple_accum:
    // set sum (%ax) to 0
    pushq %r8
    pushq %r9
    pushq %r10
    xor %r8,%r8
    xor %r9,%r9
    xor %r10,%r10
    xor %eax, %eax
    // return immediately; special case if size (%rdi) == 0
    test %rdi, %rdi
    je .L_done
    // store i = 0 in rcx
    movq $0, %rcx
// labels starting with '.L' are local to this file
.L_loop:
    // sum (%ax) += a[i]
    addw (%rsi,%rcx,2), %ax
    // sum (%ax) += a[i+1]
    addw 2(%rsi,%rcx,2), %r8w
    addw 4(%rsi,%rcx,2), %r9w
    addw 6(%rsi,%rcx,2), %r10w
    // i += 1
    addq $4, %rcx
    // i < end?
    cmpq %rdi, %rcx
    jl .L_loop
.L_done:
    addw %r8w, %ax
    addw %r9w, %ax
    addw %r10w, %ax
    popq %r8
    popq %r9
    popq %r10
    retq

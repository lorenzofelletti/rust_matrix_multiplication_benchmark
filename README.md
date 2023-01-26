# Matrix Multiplication Benchmark

This is a simple benchmark for matrix multiplication. It is written in Rust and allows to compare the performance of different matrix multiplication algorithms. Specifically, it compares the performance of the following algorithms:

* Matrix multiplication with ijk loop order
* Matrix multiplication with ikj loop order
* Matrix multiplication with parallelized i loop (ikj loop order)
* Matrix multiplication with tilings (ikj loop order).

It is possible to configure options such as the size of the matrices, the number of threads, and the tile size.

This repository was inspired from the [Mit OCW 6.172 course](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/), lecture 1 in particular.

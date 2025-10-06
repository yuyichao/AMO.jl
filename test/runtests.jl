#!/usr/bin/julia

using Distributed

addprocs(Sys.CPU_THREADS)

pmap(["pauli",
      "atomic",
      "trap",
      "rand",
      "time_sequence",
      "math",
      "utils",
      ]) do file
          println("Start testing $file")
          @eval module $(Symbol("Test_$(file)_mod"))
          include($(joinpath(@__DIR__, "$(file).jl")))
          end
          println("Done testing $file")
          # So that we do not try to bring the worker-only module back to the main process
          return
      end

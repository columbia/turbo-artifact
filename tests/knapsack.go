package main

import (
	"C"
	"unsafe"
	"fmt"
	"reflect"
)

// Build with: go build -buildmode=c-shared -o knapsack.so knapsack.go

// TODO: Compute the whole relevance matrix in go directly.
// We have to pass a bunch of matrices, but then we can parallelize more effectively.

//export SolveApprox
func SolveApprox(capacity C.double, demands *C.double, profits *C.double, length C.int) C.double {

	// See on:
	// https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices

	demands_header := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(demands)),
		Len:  int(length),
		Cap:  int(length),
	}

	profits_header := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(profits)),
		Len:  int(length),
		Cap:  int(length),
	}

	demands_slice := *(*[]float64)(unsafe.Pointer(&demands_header))
	fmt.Printf("Demands %v\n", demands_slice)

	profits_slice := *(*[]float64)(unsafe.Pointer(&profits_header))
	fmt.Printf("Profits %v\n", profits_slice)


	result := solve(float64(capacity), profits_slice, profits_slice)

	return C.double(result)
	
}


func solve(
	capacity float64,
	demands []float64,
	profits []float64,
) float64 {


	// TODO: implement the 1/k approx algo
	fmt.Printf("Capacity: %v\n", capacity)

	return 0.3
}


func main() {
}
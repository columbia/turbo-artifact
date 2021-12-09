package main

import (
	"C"
	"unsafe"
	"fmt"
	"reflect"
	"math"
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

// Softmax returns the softmax of m with temperature T.
// i.e. exp(x / T) / sum(exp(x / T)) in vector form
func Softmax(x []float64, T float64) []float64 {
	r := make([]float64, len(x))
	d := 1e-15 // Denominator, don't divide by zero

	// Substract the max to avoid overflow
	m := 0.0
	for i, v := range x {
		if i==0 || v/T > m {
			m = v/T
		}
	}

	// Denominator
	for _, v := range x {
		d += math.Exp((v-m)/T)
	}

	// Softmax vector
	for i, v := range x {
		r[i] = math.Exp((v-m)/T) / d
	}

	return r
}


func main() {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{-1, 0, 5}
	z := []float64{0.1, 1, 1000, 50}
	w := []float64{5}
	v := []float64{0}

	T := 10.0
	fmt.Println(Softmax(x, T))
	fmt.Println(Softmax(x, 1))
	fmt.Println(Softmax(y, 1))
	fmt.Println(Softmax(z, 1))
	fmt.Println(Softmax(w, 1))
	fmt.Println(Softmax(v, 1))


}
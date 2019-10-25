package main

import (
	"fmt"

	"github.com/c0z0/gotorch/autograd"
	"github.com/c0z0/gotorch/tensor"
)

func main() {
	th1 := autograd.CreateVariable(tensor.CreateScalar(0.1), "th1")
	th2 := autograd.CreateVariable(tensor.CreateScalar(0.1), "th2")

	lr := autograd.CreateVariable(tensor.CreateScalar(.01), "lr")

	X := []*autograd.Variable{autograd.CreateVariable(tensor.CreateScalar(0), "X1"), autograd.CreateVariable(tensor.CreateScalar(2), "X2")}
	Y := []*autograd.Variable{autograd.CreateVariable(tensor.CreateScalar(1), "Y1"), autograd.CreateVariable(tensor.CreateScalar(3), "Y2")}

	for e := 0; e < 1000; e++ {
		YH := X[e%2].Mul(th1).Add(th2)
		L := Y[e%2].Sub(YH).Pow(2)

		L.Backward()

		th1 = th1.Sub(autograd.CreateVariable(th1.Grad(), "").Mul(lr))
		th2 = th2.Sub(autograd.CreateVariable(th2.Grad(), "").Mul(lr))

		fmt.Println(L.Data())
	}

	fmt.Println(th1.Data())
	fmt.Println(th2.Data())

}

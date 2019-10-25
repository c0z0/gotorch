package autograd

import (
	"github.com/c0z0/gotorch/tensor"
)

const (
	addOp = iota
	subOp = iota
	mulOp = iota
	powOp = iota
)

type operation struct {
	operation int
	childA    *Variable
	childB    *Variable
	power     int
}

func (o *operation) passGrad(g tensor.Scalar) {
	switch o.operation {
	case addOp:
		{
			o.childA.passGrad(g)
			o.childB.passGrad(g)
		}
	case mulOp:
		{
			o.childA.passGrad(g.Mul(o.childB.tensor))
			o.childB.passGrad(g.Mul(o.childA.tensor))
		}
	case powOp:
		{
			o.childA.passGrad(g.Mul(tensor.CreateScalar((float64)(o.power)).Mul(o.childA.tensor.Pow(o.power - 1))))
		}
	case subOp:
		{
			o.childA.passGrad(g)
			o.childB.passGrad(g.Mul(tensor.CreateScalar(-1)))
		}
	}
}

type Variable struct {
	tensor           tensor.Scalar
	grad             tensor.Scalar
	previosOperation *operation
	debugLabel       string
	requiresGrad     bool
}

func CreateVariable(t tensor.Scalar, label string) *Variable {
	return &Variable{tensor: t, requiresGrad: true, debugLabel: label}
}

func (a *Variable) Add(b *Variable) *Variable {
	return &Variable{tensor: a.tensor.Add(b.tensor), previosOperation: &operation{childA: a, childB: b, operation: addOp}}
}

func (a *Variable) Sub(b *Variable) *Variable {
	return &Variable{tensor: a.tensor.Sub(b.tensor), previosOperation: &operation{childA: a, childB: b, operation: subOp}}
}

func (a *Variable) Mul(b *Variable) *Variable {
	return &Variable{tensor: a.tensor.Mul(b.tensor), previosOperation: &operation{childA: a, childB: b, operation: mulOp}}
}

func (a *Variable) Pow(p int) *Variable {
	return &Variable{tensor: a.tensor.Pow(p), previosOperation: &operation{childA: a, operation: powOp, power: p}}
}

func (a Variable) Data() float64 {
	return a.tensor.Data()
}

func (a Variable) Grad() tensor.Scalar {
	return a.grad
}

func (a *Variable) SetData(d float64) {
	a.tensor.SetData(d)
}

func (v *Variable) passGrad(g tensor.Scalar) {
	if v.requiresGrad {
		v.grad = g
	}

	if v.previosOperation != nil {
		v.previosOperation.passGrad(g)
	}
}

func (v *Variable) Backward() {
	v.passGrad(tensor.CreateScalar(1))
}

package tensor

type Scalar struct {
	data float64
}

func pow(f float64, p int) float64 {
	if p == 0 {
		return 1
	}

	return f * pow(f, p-1)
}

func CreateScalar(data float64) Scalar {
	return Scalar{data: data}
}

func (a Scalar) Mul(b Scalar) Scalar {
	return Scalar{data: a.data * b.data}
}

func (a Scalar) Add(b Scalar) Scalar {
	return Scalar{data: a.data + b.data}
}

func (a Scalar) Sub(b Scalar) Scalar {
	return Scalar{data: a.data - b.data}
}

func (a Scalar) Pow(p int) Scalar {
	return Scalar{data: pow(a.data, p)}
}

func (a Scalar) Data() float64 {
	return a.data
}

func (a *Scalar) SetData(d float64) {
	a.data = d
}

package tc

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

type rec struct {
	cl int
	fv []float64
}

func argmax(x []float64) int {
	j := 0
	max := 0.0
	for i := 0; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
			j = i
		}
	}
	return j
}

func TestStudent(t *testing.T) {
	x := []rec{ // класс, [очки, галстук, билет]
		// студент
		{0, []float64{1.0, 1.0, 1.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		{0, []float64{1.0, 1.0, 0.0}},
		// не студент
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{0.0, 1.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
		{1, []float64{1.0, 0.0, 0.0}},
	}

	c := New(2, 3)

	for _, r := range x {
		err := c.Learn(r.cl, r.fv)
		if err != nil {
			t.Fatal(err)
		}
	}

	// генерация новой выборки по частотам тренировочной
	rand.Seed(time.Now().UnixNano())
	cnt := 1000000
	y := make([]rec, cnt)
	x0 := x[:4]
	x1 := x[4:]
	for i := 0; i < cnt; i++ {
		r := x[rand.Intn(len(x))]
		y[i].cl = r.cl
		y[i].fv = make([]float64, 3)
		if r.cl == 0 {
			y[i].fv[0] = x0[rand.Intn(len(x0))].fv[0]
			y[i].fv[1] = x0[rand.Intn(len(x0))].fv[1]
			y[i].fv[2] = x0[rand.Intn(len(x0))].fv[2]
		} else {
			y[i].fv[0] = x1[rand.Intn(len(x1))].fv[0]
			y[i].fv[1] = x1[rand.Intn(len(x1))].fv[1]
			y[i].fv[2] = x1[rand.Intn(len(x1))].fv[2]
		}
	}

	for _, sample := range [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{1.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{1.0, 0.0, 1.0},
		{0.0, 1.0, 1.0},
		{1.0, 1.0, 1.0},
	} {
		fmt.Println("case: ", sample)

		p, err := c.Predict(sample)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("[%.4f, %.4f]\n", p[0], p[1])

		// проверка экспериментом
		total := 0.0
		t0 := 0.0
		t1 := 0.0
		for i := 0; i < cnt; i++ {
			// все признаки должны совпадать
			if (sample[0] == y[i].fv[0]) && (sample[1] == y[i].fv[1]) && (sample[2] == y[i].fv[2]) {
				total++
				if y[i].cl == 0 {
					t0++
				}
				if y[i].cl == 1 {
					t1++
				}
			}
		}
		p0 := 0.0
		p1 := 0.0
		if total > 0 {
			p0 = t0 / total
			p1 = t1 / total
		}
		fmt.Printf("[%.4f, %.4f]\n", p0, p1)
	}
}

package tc

import "errors"

type Classifier struct {
	cc int       // classes count
	fc int       // features count
	fs []float64 // feature statistics by class
	ft []float64 // features total
	ct []float64 // classes total
}

func New(classes, features int) *Classifier {
	return &Classifier{
		cc: classes,
		fc: features,
		fs: make([]float64, classes*features),
		ft: make([]float64, features),
		ct: make([]float64, classes),
	}
}

func (c *Classifier) Learn(class int, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
	c.ct[class] += 1
	base := class * c.fc
	fc := len(fv)
	if fc > c.fc {
		fc = c.fc
	}
	for fi := 0; fi < fc; fi++ {
		if fv[fi] < 0 || fv[fi] > 1 {
			return errors.New("feature value must be in range 0..1")
		}
		c.fs[base+fi] += fv[fi]
		c.ft[fi] += fv[fi]
	}
	return nil
}

func (c *Classifier) Predict(fv []float64) ([]float64, error) {
	p := make([]float64, c.cc)
	t := 0.0
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cp := 1.0                    // calculated class probability
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			if fv[fi] < 0 || fv[fi] > 1 {
				return nil, errors.New("feature value must be in range 0..1")
			}
			if c.ct[ci] == 0 {
				continue
			}
			if c.ft[fi] == 0 {
				continue
			}
			fp := cf[fi] / c.ct[ci] // feature probability in this class; we can calculate this in advance, but it will take more memory (+classes*features*float64)
			if fv[fi] == 0 {
				fp = 1 - fp
			}
			// fp = fp * fv[fi] // feature value limited by range 0 <= v <= 1, so we just reduce probability proportionally
			cp = cp * fp // probability of combination of features in this class
		}
		p[ci] = cp * c.ct[ci] // scale proportionally to probability of this class
		t += p[ci]
	}
	if t > 0 {
		for i := 0; i < len(p); i++ {
			p[i] = p[i] / t // probability in relation to other classes
		}
	}
	return p, nil
}

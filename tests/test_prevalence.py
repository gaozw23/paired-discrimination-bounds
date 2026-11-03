"""Direct-pair, independently enumerated regression checks for continuous hulls."""
from fractions import Fraction as F
import json
import math

import numpy as np
import pytest

from paired_auc import paired_bounds_exact_count
from paired_auc.a2 import binomial_plan, clopper_pearson
from paired_auc.prevalence import Algebraic, EmpiricalPrevalence, InputError
from tests.prevalence_oracle import binary_completions, global_oracle


CASES = [
    ([0,0,2,3,3],[3,1,1,0,4],[0,0,0,1,1],[0,0,0,0,1]),
    ([7,1,2,5,3,6,4,8],[7,1,3,6,8,4,2,5],[1,0,1,0,1,1,0,0],[1,0,0,0,0,1,0,0]),
    ([0,1,2,3],[3,2,1,0],[0,0,0,0],[0,0,0,0]),
    ([0,0,0,0],[0,0,0,0],[1,0,0,1],[0,0,0,1]),
    ([0,1,2,3],[3,1,2,0],[1,1,1,1],[1,0,1,0]),
]


@pytest.mark.parametrize('a,b,r,y',CASES)
def test_finite_bridge_independent_direct_pair_completions(a,b,r,y):
    observed = np.where(r,y,np.nan)
    law = EmpiricalPrevalence(a,b,r,observed)
    for m, group in binary_completions(a,b,r,y).items():
        if m in (0,len(a)):
            continue
        values = [float(value) for value,_ in group]
        continuous = law.envelope([[m/len(a),m/len(a)]])
        finite = paired_bounds_exact_count(a,b,observed,r,m)
        assert continuous['status'] == 'valid'
        for actual,expected in [(continuous['lower'],min(values)),(continuous['upper'],max(values)),
                                (finite.lower,min(values)),(finite.upper,max(values))]:
            assert actual == pytest.approx(expected,abs=1e-12,rel=1e-12)


@pytest.mark.parametrize('a,b,r,y',CASES)
@pytest.mark.parametrize('intervals',[[[0,1]],[[.15,.8]],[[0,.1],[.4,.6],[.9,1]],[[0,0]],[[1,1]]])
def test_continuous_global_decimal_oracle(a,b,r,y,intervals):
    expected = global_oracle(a,b,r,y,intervals)
    actual = EmpiricalPrevalence(a,b,r,np.where(r,y,np.nan)).envelope(intervals)
    if expected['status'] == 'empty':
        assert actual['status'] == 'empty_domain'
    else:
        assert actual['status'] == 'valid'
        for side in ('lower','upper'):
            assert actual[side] == pytest.approx(expected[side],abs=1e-12,rel=1e-12)
            assert actual[side+'_attainment']['attained'] == expected[side+'_attained']
    json.dumps(actual,allow_nan=False)


def test_no_hidden_labels_and_no_epsilon_snapping():
    with pytest.raises(InputError):
        EmpiricalPrevalence([1,2],[2,1],[1,0],[0,1])
    law = EmpiricalPrevalence([1,2,3,4,5],[5,4,3,2,1],[1,1,1,1,0],[1,0,0,0])
    assert law.diagnose_prevalence(.2)['feasible']
    assert not law.diagnose_prevalence(math.nextafter(.2,-math.inf))['feasible']


def test_exact_algebraic_order_and_cancellation():
    assert Algebraic(F(1),F(2),2) == Algebraic(F(1),F(1),8)
    assert Algebraic(F(-10**60),F(1),10**120+1) > 0
    assert Algebraic(F(10**60),F(-1),10**120+1) < 0
    assert Algebraic(F(1),F(1),2) < Algebraic(F(1)+F(1,10**100),F(1),2)


def test_interior_stationary_extreme_is_not_an_integer_grid_hull():
    a = [6,7,5,6,7,0,2,4,3,0]
    b = [6,3,3,2,4,0,2,4,7,7]
    r = [1,1,0,1,0,1,1,0,1,1]
    y = [0,1,0,1,0,0,0,0,0,0]
    result = EmpiricalPrevalence(a,b,r,np.where(r,y,np.nan)).envelope([[.2,.5]])
    expected = global_oracle(a,b,r,y,[[.2,.5]])
    assert result['lower'] == pytest.approx(expected['lower'],abs=1e-12,rel=1e-12)
    assert any(x['kind']=='stationary' for x in result['lower_attainment']['candidates'])
    binary = binary_completions(a,b,r,y)
    binary_min = min(float(value) for group in binary.values() for value,_ in group if value is not None)
    assert result['lower'] < binary_min - 1e-6
    assert result['diagnostics']['max_scaled_root_residual'] <= 1e-12


def test_excluded_endpoint_limit_is_not_reported_as_attained():
    law = EmpiricalPrevalence([0,1],[1,0],[1,0],[0])
    result = law.envelope([[0,1]])
    assert result['lower_attainment']['attained'] is False
    assert result['lower_attainment']['limits'] == [0.0]
    assert result['upper_attainment']['attained'] is True


@pytest.mark.parametrize('n,truth',[(200,.2),(200,.5),(1000,.2),(1000,.5)])
def test_external_all_k_raw_weights_cp_and_underflow(n,truth):
    rows,check = binomial_plan(n,truth)
    assert len(rows) == n+1
    assert abs(check['raw_mass_error']) <= 1e-12
    assert check['renormalized'] is False
    assert rows[0]['cp_lower'] == 0 and rows[-1]['cp_upper'] == 1
    assert all(0 <= row['cp_lower'] <= row['cp_upper'] <= 1 for row in rows)
    if n==1000 and truth==.2:
        assert check['underflow_count'] > 0
        assert math.isfinite(check['underflow_log_mass'])
    assert clopper_pearson(0,n) == (rows[0]['cp_lower'],rows[0]['cp_upper'])

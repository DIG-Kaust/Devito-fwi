import pytest

from package.add import add

par1 = {'x': 10, 'y': 4} 
par2 = {'x': -2, 'y': 2}

@pytest.mark.parametrize("par", [(par1), (par2)])
def test_add(par):
    x, y = par['x'], par['y']    
    assert add(x,y) == x + y

import pytest
from hml_fixtures import device_type, has_cuda, dtype, to_np_dtype
from hml_fixtures import mp
import numpy as np
from itertools import permutations


class TestTensorShapeTransfrom(object):
    def test_reshape(self):
        a = mp.arange(0, 100).reshape((2, 5, 5, 2))
        assert(a.is_contiguous)

        a = a.slice(1, 1, 5, 2)
        assert(a.is_contiguous == False)
        
        b = a.reshape((2, -1, 10))
        assert(b.is_contiguous == False)

        b = a.reshape((1, 4, 10)) # clone, and reshape
        assert(b.is_contiguous == True)

        b = a.reshape((2, 4, 5)) # clone, and reshape
        assert(b.is_contiguous == True)


    def test_select_slice(self):
        a = mp.arange(0, 1000).reshape((2, 4, 5, 5, 5))
        b = np.arange(0, 1000).reshape((2, 4, 5, 5, 5))
        assert((a.numpy() == b).all())

        c = a.select(0, 1).select(1, 2)
        d = b[1, :, 2, :, :]
        assert((c.numpy() == d).all())

        c = a.slice(1, 1, 4).slice(1, 0, 3, 2).slice(3, 2, 5, 2)
        d = b[:, 1:4:2, :, 2:5:2, :]
        assert((c.numpy() == d).all())

        c = a.select(0, 1).slice(2, 1, 4, 2).select(1, 2)
        d = b[1, :, 2, 1:4:2, :]
        assert((c.numpy() == d).all())

        #negetive indexing
        c = a.select(-5, -1).slice(2, -4, -1, 2).select(-3, 2)
        d = b[-1, :, 2, 1:4:2, :]
        assert((c.numpy() == d).all())


    def test_transpose(self):
        a = mp.arange(0, 1000).reshape((2, 4, 5, 5, 5))
        b = a.cpu().numpy()

        for p in permutations(range(len(a.shape)), 2):
            pp = list(range(len(a.shape)))
            pp[p[0]] = p[1]
            pp[p[1]] = p[0]
            c = a.transpose(p[0], p[1])
            d = np.transpose(b, pp)
            assert((c.numpy() == d).all())

        # negetive index
        p = [-1, -3]
        pp = list(range(len(a.shape)))
        pp[p[0]] = p[1]
        pp[p[1]] = p[0]
        c = a.transpose(p[0], p[1])
        d = np.transpose(b, pp)
        assert((c.numpy() == d).all())


    def test_permute(self):
        a = mp.arange(0, 1000).reshape((2, 4, 5, 5, 5))
        b = a.cpu().numpy()

        for p in permutations(range(len(a.shape))):
            c = a.permute(p)
            d = np.transpose(b, p)
            assert((c.numpy() == d).all())

        # negetive index
        p = [-1, -2, -3, -4, -5]
        c = a.permute(p)
        d = np.transpose(b, p)
        assert((c.numpy() == d).all())


    def test_concat(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6]])
        c = np.concatenate((a, b), axis=0)
        d = np.concatenate((a, b.T), axis=1) 

        mp_a = mp.from_numpy(a)
        mp_b = mp.from_numpy(b)

        # non-contiguous
        mp_c = mp.concat((mp_a, mp_b), axis=0)
        mp_d = mp.concat((mp_a, mp_b.transpose(0, 1)), axis=1) 

        # out and negtive axis
        mp_d2 = mp.empty_like(mp_d, device=mp_d.device, dtype=mp_d.dtype)
        mp.concat(mp_d2, (mp_a, mp_b.transpose(0, 1)), axis=-1)

        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())
        assert((mp_d2.numpy() == d).all())

        # shape not match expect axis
        with pytest.raises(RuntimeError):
            mp.concat((mp_a, mp_b), axis=1)


    def test_stack(self):
        a = [np.random.randn(3, 4) for _ in range(10)]
        b = np.stack(a, axis=0)
        c = np.stack(a, axis=1)
        d = np.stack(a, axis=2)

        mp_a = [mp.from_numpy(v) for v in a]
        mp_b = mp.stack(mp_a, axis=0)
        mp_c = mp.stack(mp_a, axis=1)
        mp_d = mp.stack(mp_a, axis=-1)
        mp_d2 = mp.empty_like(mp_d, device=mp_d.device, dtype=mp_d.dtype)
        mp.stack(mp_d2, mp_a, axis=-1) #out

        assert((mp_b.numpy() == b).all())
        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())
        assert((mp_d2.numpy() == d).all())

        # shape not match
        mp_e = mp.arange(4).reshape((2, -1))
        mp_f = mp.arange(6).reshape((2, -1))
        with pytest.raises(RuntimeError):
            mp.stack((mp_e, mp_f), axis=0)

    
    def test_vstack(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.vstack((a, b))

        e = np.array([[1], [2], [3]])
        f = np.array([[2], [3], [4]])
        d = np.vstack((e, f))

        mp_a = mp.from_numpy(a)
        mp_b = mp.from_numpy(b)
        mp_e = mp.from_numpy(e)
        mp_f = mp.from_numpy(f)
        mp_c = mp.vstack((mp_a, mp_b))
        mp_d = mp.vstack((mp_e, mp_f))
        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())


    def test_hstack(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.hstack((a, b))

        e = np.array([[1], [2], [3]])
        f = np.array([[2], [3], [4]])
        d = np.hstack((e, f))

        mp_a = mp.from_numpy(a)
        mp_b = mp.from_numpy(b)
        mp_e = mp.from_numpy(e)
        mp_f = mp.from_numpy(f)
        mp_c = mp.hstack((mp_a, mp_b))
        mp_d = mp.hstack((mp_e, mp_f))
        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())


    def test_flatten(self):
        a = np.arange(12).reshape((3, 4))
        b = a[::2]

        c = a.flatten()
        d = b.flatten()

        mp_a = mp.from_numpy(a)
        mp_b = mp_a.slice(0, 0, 3, 2)
        mp_c = mp_a.flatten()
        mp_d = mp_b.flatten()

        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())


    def test_squeeze_unsqueeze(self):
        a = np.arange(24).reshape((6, 4))[::2, :]
        b = a[np.newaxis, ...]
        c = a.reshape((3, 1, 4))
        d = a[..., np.newaxis]

        mp_a = mp.from_numpy(a)
        mp_b = mp_a.unsqueeze(0)
        mp_c = mp_a.unsqueeze(1)
        mp_d = mp_a.unsqueeze(2)
        mp_e = mp_a.unsqueeze(1).unsqueeze(2)

        mp_f = mp_b.squeeze(0)
        mp_g = mp_c.squeeze(1)
        mp_h = mp_d.squeeze(2)
        mp_i = mp_e.squeeze(2).squeeze(1)

        assert((mp_b.numpy() == b).all())
        assert((mp_c.numpy() == c).all())
        assert((mp_d.numpy() == d).all())
        assert((mp_f.numpy() == a).all())
        assert((mp_g.numpy() == a).all())
        assert((mp_h.numpy() == a).all())
        assert((mp_i.numpy() == a).all())

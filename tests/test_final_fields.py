import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from geobc.final_fields import read_final_fields,LOADS


class FinalReadout(unittest.TestCase):
    def test_barrier_precedes_file_access(self):
        with self.assertRaises(RuntimeError):
            read_final_fields(Path('absent.csv'),np.array([0],dtype=np.int64),np.array([1]),False)

    def test_grid_alignment_signed_fields_and_excluded_columns(self):
        header=['nodeID']+[load+suffix for load in LOADS for suffix in ['_x_disp(mm)','_y_disp(mm)','_z_disp(mm)','_stress(MPa)']]+['dia_stress(MPa)']
        with TemporaryDirectory() as folder:
            path=Path(folder)/'fields.csv'
            path.write_text(','.join(header)+'\n'+','.join(map(str,[40]+[1,2,3,-4]*3))+',unparsed\n'+','.join(map(str,[30]+[5,6,7,-8]*3))+',unparsed\n')
            result=read_final_fields(path,np.array([1,0],dtype=np.int64),np.array([30,40],dtype=np.int64),True)
            for load in LOADS:
                self.assertEqual(result[load].dtype,np.float32)
                np.testing.assert_array_equal(result[load],[[5,6,7,-8],[1,2,3,-4]])


if __name__=='__main__':unittest.main()

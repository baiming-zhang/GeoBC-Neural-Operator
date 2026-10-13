import json
import unittest
from geobc.deployment import handle_line,validate_request


class RequestContract(unittest.TestCase):
    def test_unsupported_input_rejected_before_backend_access(self):
        class Session:
            catalog={'cases':{'registered':{}}}
            def query(self,request):
                validate_request(request,{'registered'})
                raise AssertionError('Invalid request accessed geometry')
        for request in [dict(cad='unknown',load='ver',xyz_mm=[[1,2,3]]),dict(cad='registered',load='new_load',xyz_mm=[[1,2,3]]),dict(cad='registered',load='ver',xyz_mm=[[1,2,float('nan')]]),dict(cad='registered',load='ver',xyz_mm=[[True,2,3]])]:
            result=handle_line(Session(),json.dumps(request))
            self.assertEqual(result['status'],'invalid_request');self.assertIsNone(result['U_mm']);self.assertIsNone(result['signed_S_MPa'])

    def test_malformed_json_and_element_id_inputs_rejected(self):
        class Session:
            catalog={'cases':{'registered':{}}}
            def query(self,request):
                validate_request(request,{'registered'})
                raise AssertionError('Invalid request accessed backend')
        for line in ['{',json.dumps(dict(cad='registered',load='ver',xyz_mm=[[1,2,3]],cell_rows=[0]))]:
            self.assertEqual(handle_line(Session(),line)['status'],'invalid_request')


if __name__=='__main__':unittest.main()

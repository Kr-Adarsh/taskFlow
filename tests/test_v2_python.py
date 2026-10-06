import hashlib
import json
import pytest
from backend.app.capabilities.python import PythonCapability
from backend.app.capabilities.python.profile import dataset_profile
from backend.app.capabilities.python.safety import validate_code
from backend.app.capabilities.python.sandbox import isolated_stage
from backend.app.capabilities.python.verification import CalculationContract, calculate
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.seed import reset_demo_env


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    reset_demo_env()
    monkeypatch.setenv('TASKFLOW_ARTIFACTS_DIR',str(tmp_path/'artifacts'))
    path=tmp_path/'input.csv'
    rows=['market,month,volume']+[f'Alpha,2026-08,{10+i}' for i in range(30)]+[f'Alpha,2026-09,{9+i}' for i in range(30)]+['Beta,2026-08,100','Beta,2026-09,1']
    path.write_text('\n'.join(rows)+'\n')
    with get_db_connection() as conn:
        conn.execute("INSERT INTO documents_index(filename,filepath,title,doc_type,created_at) VALUES(?,?,?,?,?)",('input.csv',str(path),'Local input','dataset','now'));conn.commit()
    return path


def test_profile_is_bounded_with_provenance(dataset):
    profile=dataset_profile('input.csv')
    assert profile['shape']==[62,3] and len(profile['sample_rows'])==3
    assert profile['sha256']==hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert 'Beta' in profile['categorical_values']['market']
    assert len(json.dumps(profile))<4000


@pytest.mark.parametrize('code',["import os", "from pandas import *", "pd.read_pickle(inputs['input.csv'])", "getattr(pd,'x')", "pd.__dict__", "eval('1')"])
def test_unsafe_code_rejected(code):
    with pytest.raises(ValueError):validate_code(code)


@pytest.mark.anyio
async def test_sandbox_executes_full_data_and_tracks_artifacts(dataset):
    capability=PythonCapability(run_id='sandbox_full')
    assert capability.profile_dataset('input.csv').ok
    code="""df=pd.read_csv(inputs['input.csv'])
totals=df.groupby(['market','month'])['volume'].sum().unstack()
difference=totals['2026-08']-totals['2026-09']
result={'summary':'Compared periods','metrics':{'market':str(difference.idxmax()),'decline':float(difference.max()),'rows':int(len(df))},'tables':{'totals':totals.reset_index()}}
"""
    result=await capability.execute_python(['input.csv'],code)
    assert result.ok,result.error
    assert result.data['metrics']=={'market':'Beta','decline':99.0,'rows':62}
    assert result.data['stage']=='full' and result.data['isolation']['seccomp']
    assert len(result.data['artifacts'])==1
    assert result.data['inputs'][0]['rows']==62
    assert result.data['tables']['totals']['preview'] and result.data['artifacts'][0]['sha256']


@pytest.mark.anyio
async def test_three_program_attempts_and_no_reset_by_reprofiling(dataset):
    capability=PythonCapability(run_id='repair')
    capability.profile_dataset('input.csv')
    failed=await capability.execute_python(['input.csv'],"result={'metrics':{'x':unknown_name}}")
    assert not failed.ok and 'NameError' in failed.error
    repaired=await capability.execute_python(['input.csv'],"result={'summary':'Rows counted','metrics':{'rows':len(pd.read_csv(inputs['input.csv']))}}")
    assert repaired.ok,repaired.error
    failed=await capability.execute_python(['input.csv'],"result={'metrics':{'x':another_unknown_name}}")
    assert not failed.ok
    capability.profile_dataset('input.csv')
    exhausted=await capability.execute_python(['input.csv'],"result={'metrics':{'x':1}}")
    assert exhausted.error_code=='CODE_REPAIR_EXHAUSTED'


@pytest.mark.anyio
async def test_input_is_read_only_and_host_credentials_absent(dataset,monkeypatch):
    monkeypatch.setenv('GROQ_KEY','nonsecret-test-marker')
    capability=PythonCapability(run_id='boundaries')
    capability.profile_dataset('input.csv')
    before=dataset.read_bytes()
    denied=await capability.execute_python(['input.csv'],"pd.DataFrame({'x':[1]}).to_csv(inputs['input.csv'],index=False)\nresult={'metrics':{}}")
    assert not denied.ok and dataset.read_bytes()==before
    result=await capability.execute_python(['input.csv'],"result={'summary':'Checked environment','metrics':{'key_present':bool(pd.io.common.os.environ.get('GROQ_KEY'))}}")
    assert result.ok,result.error
    assert result.data['metrics']['key_present'] is False


@pytest.mark.anyio
async def test_sandbox_timeout_and_network_subprocess_denial(dataset,tmp_path):
    timeout=await isolated_stage('while True: pass',{'input.csv':dataset},tmp_path/'timeout','full',timeout=1)
    assert not timeout['ok'] and timeout['error_code']=='PYTHON_TIMEOUT'
    denied=await isolated_stage("pd.io.common.os.posix_spawn('/usr/bin/true',['true'],{})\nresult={'metrics':{}}",{'input.csv':dataset},tmp_path/'spawn','full')
    assert not denied['ok'] and 'denied' in denied['error'].lower()
    network=await isolated_stage("pd.read_csv('http://127.0.0.1:8000/private')\nresult={'metrics':{}}",{'input.csv':dataset},tmp_path/'network','full')
    assert not network['ok']


def test_independent_calculation_uses_full_rows_and_rejects_ties(dataset):
    contract=CalculationContract(document_id='input.csv',group_column='market',value_column='volume',period_column='month',baseline_period='2026-08',current_period='2026-09',measure='difference',convention='baseline_minus_current',selection='max',group_metric='market',value_metric='decline')
    expected=calculate(contract)
    assert expected['group']=='Beta' and expected['value']==99 and expected['full_dataset_rows']==62


@pytest.mark.anyio
async def test_changed_input_requires_new_profile(dataset):
    capability=PythonCapability(run_id='changed');capability.profile_dataset('input.csv')
    dataset.write_text(dataset.read_text()+'Gamma,2026-08,1\n')
    result=await capability.execute_python(['input.csv'],"result={'metrics':{}}")
    assert not result.ok and 'changed after profiling' in result.error


@pytest.mark.anyio
async def test_sandbox_memory_stdout_and_output_bounds(dataset,tmp_path):
    memory=await isolated_stage("value=np.ones(200000000)\nresult={'metrics':{'value':1}}",{'input.csv':dataset},tmp_path/'memory','full')
    assert not memory['ok']
    output=await isolated_stage("print('x'*20000)\nresult={'metrics':{'value':1}}",{'input.csv':dataset},tmp_path/'stdout','full')
    assert output['ok'] and len(output['stdout'])<=4096
    huge=await isolated_stage("result={'metrics':{'raw':list(range(10000))}}",{'input.csv':dataset},tmp_path/'huge','full')
    assert not huge['ok'] and 'scalar' in huge['error']
    disk=await isolated_stage("pd.io.common.os.mkdir('/dev/extra')\nresult={'metrics':{}}",{'input.csv':dataset},tmp_path/'device','full')
    assert not disk['ok']


@pytest.mark.anyio
async def test_no_execution_fallback_when_namespace_tool_unavailable(dataset,tmp_path,monkeypatch):
    monkeypatch.setattr('backend.app.capabilities.python.sandbox.shutil.which',lambda _:None)
    result=await isolated_stage("result={'metrics':{'value':1}}",{'input.csv':dataset},tmp_path/'unavailable','full')
    assert not result['ok'] and result['error_code']=='SANDBOX_UNAVAILABLE'


@pytest.mark.anyio
async def test_native_stdout_flood_is_killed_without_unbounded_capture(dataset,tmp_path):
    code="while True: pd.io.common.os.write(1,b'x'*1048576)"
    result=await isolated_stage(code,{'input.csv':dataset},tmp_path/'flood','full')
    assert not result['ok'] and result['error_code']=='OUTPUT_LIMIT'


def test_context_profile_keeps_schema_but_bounds_examples_and_summaries(dataset):
    from backend.app.capabilities.python.profile import context_profile
    profile=dataset_profile('input.csv')
    compact=context_profile(profile,'Compare volume by market')
    assert compact['column_types']==profile['dtypes']
    assert len(json.dumps(compact))<=8000 and len(compact['sample_rows'])==3
    assert 'volume' in compact['numeric_summaries']


def test_verification_does_not_return_all_group_rows(dataset):
    # Full calculation may compare many groups; only bounded evidence leaves it.
    dataset.write_text('market,volume\n'+'\n'.join(f'Group{i},{i}' for i in range(1000)))
    contract=CalculationContract(document_id='input.csv',group_column='market',value_column='volume',aggregate='sum',measure='value',selection='max',group_metric='market',value_metric='volume')
    result=calculate(contract)
    assert result['group']=='Group999' and result['groups_compared']==1000
    assert len(result['group_values_preview'])==10 and len(json.dumps(result))<1000

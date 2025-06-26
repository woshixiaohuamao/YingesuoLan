from datetime import datetime, timedelta
import requests
import json

def fetch_realtime_data():
    """获取流量数据的实时数据"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)

    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,   # 1分钟间隔
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_LLJ01_FQ01.PV,DLDZ_AVS_LLJ01_FQ01.PV"
    }

    sum_diffs = []

    try:
        response = requests.get(
            "",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        api_data = response.json()
        if api_data.get("code") != 0:
            raise Exception(f"API返回错误码: {api_data.get('code')}")
        
        items = api_data.get("items", [])
        field1_vals = []
        field2_vals = []
        
        for item in items:
            name = item.get("name")
            vals = item.get("vals", [])
            
            if not vals:
                raise Exception(f"字段 {name} 返回空数据")
                
            sorted_vals = sorted(vals, key=lambda x: x["time"])
            
            if name == "DLDZ_DQ200_LLJ01_FQ01.PV":
                field1_vals = [v["val"] for v in sorted_vals]
            elif name == "DLDZ_AVS_LLJ01_FQ01.PV":
                field2_vals = [v["val"] for v in sorted_vals]
        
        if len(field1_vals) != len(field2_vals):
            raise Exception(f"字段长度不一致：{len(field1_vals)} vs {len(field2_vals)}")
        
        for i in range(1, len(field1_vals)):
            diff1 = field1_vals[i] - field1_vals[i-1]
            diff2 = field2_vals[i] - field2_vals[i-1]
            sum_diffs.append(diff1 + diff2)
            
    except Exception as e:
        raise Exception(f"处理失败: {str(e)}")
    return sum_diffs[-144:]

def fetch_realtime_data_for_P():
    """获取压力数据的实时数据"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)

    params = {
        "startTime": start_time.isoformat(timespec='milliseconds'),
        "endTime": end_time.isoformat(timespec='milliseconds'),
        "interval": 60000,
        "valueonly": 0,
        "decimal": 2,
        "names": "DLDZ_DQ200_SYSTEM_PI05.PV,DLDZ_AVS_SYSTEM_PI05.PV"
    }

    try:
        response = requests.get(
            "",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        api_data = response.json()
        if api_data.get("code") != 0:
            return None

        field1_vals = []
        field2_vals = []
        time_index = {}

        for item in api_data.get("items", []):
            sorted_vals = sorted(item["vals"], key=lambda x: x["time"])
            if item["name"] == "DLDZ_DQ200_SYSTEM_PI05.PV":
                field1_vals = [(v["time"], v["val"]) for v in sorted_vals]
                time_index = {ts: idx for idx, (ts, _) in enumerate(field1_vals)}
            elif item["name"] == "DLDZ_AVS_SYSTEM_PI05.PV":
                field2_vals = [(v["time"], v["val"]) for v in sorted_vals]

        sum_vals = []
        for ts, val1 in field1_vals:
            if ts in time_index:
                val2 = next((v for t, v in field2_vals if t == ts), None)
                if val2 is not None:
                    sum_vals.append(val1 + val2)

        return sum_vals[-144:]

    except Exception as e:
        print(f"获取历史数据失败: {str(e)}")
        return None
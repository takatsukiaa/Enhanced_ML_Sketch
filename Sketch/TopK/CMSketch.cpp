#include "CMSketch.h"
using namespace std;


CMSketch::CMSketch(uint d, uint w):d(d), w(w){
	sketch = new uint*[d];
	ov_flags = new uchar*[d];
	for(uint i = 0; i < d; ++i){
		sketch[i] = new uint[w];
		ov_flags[i] = new uchar[w];
		memset(sketch[i], 0, w * sizeof(uint));
		memset(ov_flags[i], 0, w * sizeof(uchar));
	}
	hf = new HashFunction();
	para = new float[d];
	mean = new float[d];
	scale = new float[d];
	t = new uint[d];
}

CMSketch::~CMSketch(){
	for(uint i = 0; i < d; ++i) delete [] sketch[i];
	delete [] sketch;
	delete hf;
	delete [] para;
	delete [] mean;
	delete [] scale;
	delete [] t;
}

void CMSketch::Insert(cuc *str){
    for (uint i = 0; i < d; ++i){
        uint cid = hf->Str2Int(str, i) % w;
        if (sketch[i][cid] == -1) {
            return;
        }
		++sketch[i][cid];
	}
}
void CMSketch::Enhanced_Insert(cuc* str)
{
	uint my_min = 0;
	uint cid[4];
	for(uint i=0; i < d; i++)
	{
		cid[i] = hf->Str2Int(str, i) % w;
		if(sketch[i][cid[i]] == -1)
		{
			return;
		}
	}
	my_min = sketch[0][cid[0]];
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == 0)
		{
			sketch[i][cid[i]] = sketch[i][cid[i]]<<22>>22;
		}
		++sketch[i][cid[i]];
		if(sketch[i][cid[i]] < my_min || sketch[i][cid[i]] == my_min)
		{
			my_min = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]]>1023 && ov_flags[i][cid[i]] == 0)
		{
			ov_flags[i][cid[i]] = 1;
		}
	}
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == '\0')
		{
			sketch[i][cid[i]] += my_min<<10;
		}
	}
}

void CMSketch::GetHashedValue(cuc *str, uint *counters)
{
	for(uint i = 0; i < d; i++)
	{
		counters[i] = hf->Str2Int(str,i) % w;
	}
}

uint CMSketch::Query(cuc *str, bool ml){
	memset(t, 0, sizeof(t)*3);

	uint Min;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t,t+d);
	if (!ml) {
		Min = t[0];
		return Min;
	}
	else {
		printf("using ml:\n");
		std::sort(t, t+d);
		uint result = Predict(t);
		//if you want a float;
		if(t[3] - t[1] <= THRESH && result>0 && result <= t[0])
		{
			return result;
		}
		return t[0];
		// if(result > t[0] || result <= 0)
		// {
		// 	return t[0];
		// }
		// else
		// {
		// 	return result;
		// }
		//if you want a integer;
		//return (uint)Predict(t);
	}
}


uint CMSketch::Enhanced_Query(cuc* str, int* feature_count)
{
    uint my_min = UINT_MAX;
    uint cid[3];
    
    // Step 1: 計算hash值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }

    // Step 2: 遍歷每一行，獲取計數器值
    for (uint i = 0; i < d; i++) {
        uint value;
        
        // 檢查溢出標誌
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，取整個 32 位值
            value = sketch[i][cid[i]];
			*feature_count +=1;
        } else {
            // 如果未溢出，僅取後 10 位值
            value = sketch[i][cid[i]] & 1023;
			*feature_count+=2;
        }
        
        // 更新最小值
        if (value < my_min) {
            my_min = value;
        }
    }

    // Step 3: 返回最小值
    return my_min;
}
void CMSketch::Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout) {
    uint cid[4];
    uint value;
    int feature_count = 0;
    //計算 Hash 值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }
	// 計算特徵數
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
           feature_count += 1;
        } else {
           feature_count += 2;
        }
    }
	fprintf(fout, ",%d", feature_count);
    //寫入檔案
    fprintf(fout, ",%u", acc_val); // 實際值
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，輸出 32 位值
            value = sketch[i][cid[i]];
            fprintf(fout, ",%u", value);
        } else {
            // 如果未溢出，輸出22位值 and 10-bit value
            uint low = sketch[i][cid[i]] & 1023;
            uint high = sketch[i][cid[i]] >> 10;
            fprintf(fout, ",%u,%u", high, low);
        }
    }
    fprintf(fout, "\n"); 
}


void CMSketch::PrintCounterFile(cuc * str, uint acc_val, FILE * fout)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	// fprintf(fout, "Actual Size: ");
	fprintf(fout, "%u ", acc_val);
	// fprintf(fout, "Counter Values:");
	for(uint i = 0; i < d-1; ++i){
		fprintf(fout, "%u ", t[i]);
	}
	fprintf(fout,"%u",t[d-1]);
	fprintf(fout, "\n");
}
// void CMSketch::PrintCounter(cuc* str, uint acc_val){
// 	for(uint i = 0; i < d; ++i){
// 		uint cid = hf->Str2Int(str, i)%w;
// 		t[i] = sketch[i][cid];
// 	}
// 	if (1) {
// 		std::sort(t, t+d);
// 		printf("%u", acc_val);
// 		for(uint i = 0; i < d; ++i){
// 			printf(" %u", t[i]);
// 		}
// 		printf("\n");
// 	}
// }
void CMSketch::LoadPara(cuc *path){
	float temp;
	FILE *file = fopen((const char*)path, "r");
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", mean+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", scale+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", para+i);
	}
}
float CMSketch::CalculateAAE(cuc * str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs((float)t[0] - acc_val);
}
float CMSketch::CalculateAAE_ML(cuc * str, uint acc_val, float query_val)
{
	return abs(query_val - acc_val);
}
float CMSketch::CalculateARE(cuc* str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs(((float)t[0] - acc_val)/acc_val);
}
float CMSketch::Predict(uint *t){
	float res = 0;
	for(uint i = 0; i < d; ++i){
		res += para[i]*((t[i]-mean[i])/scale[i]);
	}
	return res;
}
vector<int> CMSketch::GetCounter(cuc* str){
	vector<int> result;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		result.push_back(sketch[i][cid]);
	}
	return result;
}

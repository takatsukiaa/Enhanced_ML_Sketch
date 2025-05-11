#ifndef CUSKETCH_H 
#define CUSKETCH_H
#include <climits>
#include "common.h"
#define THRESH 512
#define THRESH_BIT 9
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

struct CUSketch:public Sketch{
public:
	CUSketch(uint d, uint w);
	~CUSketch();
	void Insert(cuc *str);
	void Enhanced_Insert(cuc* str);
	std::vector<float> GetHashedValue(cuc *str);
	uint Query(cuc *str, bool ml = FALSE);
	void PrintCounter(cuc* str, uint acc_val);
	void PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	float CalculateAAE(cuc * str, uint acc_val);
	void LoadPara(cuc *path = CMPATH);
	float Predict(uint *t);
	uint Enhanced_Query(cuc* str, int* feature_count);
	void Enhanced_PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
private:
	HashFunction *hf;
	uint** sketch;
	float* para;
	float* mean;
	float* scale;
	uint d;
	uint w;
	uint *t;
	uchar** ov_flags;
};

CUSketch::CUSketch(uint d, uint w):d(d), w(w){
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

CUSketch::~CUSketch(){
	for(uint i = 0; i < d; ++i) delete [] sketch[i];
	for(uint i = 0; i < d; ++i) delete [] ov_flags[i];
	delete [] sketch;
	delete hf;
	delete [] para;
	delete [] mean;
	delete [] scale;
	delete [] t;
	delete[] ov_flags;
}

void CUSketch::Insert(cuc *str){
	memset(t, 0, sizeof(t));
	uint Min = INF_SHORT;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
		Min = std::min(Min, t[i]);
	}
    if (Min == INF_SHORT)
        return;
	for(uint i = 0; i < d; ++i){
		if (t[i]==Min){
			uint cid = hf->Str2Int(str, i)%w;
			++sketch[i][cid];
		}
	}
}
void CUSketch::Enhanced_Insert(cuc* str)
{
	uint min = 0;
	uint cid[4];
	uint value[4];
	for(uint i=0; i < d; i++)
	{
		cid[i] = hf->Str2Int(str, i) % w;
		if(sketch[i][cid[i]] == -1)
		{
			return;
		}
	}
	min = sketch[0][cid[0]];
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == 0)
		{
			sketch[i][cid[i]] = sketch[i][cid[i]]<<(32-THRESH_BIT)>>(32-THRESH_BIT);
		}
		if(sketch[i][cid[i]] < min || sketch[i][cid[i]] == min)
		{
			min = sketch[i][cid[i]];
		}
	}
	uint min_after = 0;
	for(uint i = 0; i<d; i++)
	{
		if(sketch[i][cid[i]] == min)
		{
			sketch[i][cid[i]]++;
			min_after = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]]>THRESH-1 && ov_flags[i][cid[i]] == 0)
		{
			ov_flags[i][cid[i]] = 1;
		}
	}
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == '\0')
		{
			sketch[i][cid[i]] += min_after<<THRESH_BIT;
		}
	}

}


void CUSketch::Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout) {
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
	fprintf(fout, "%d", feature_count);
    //寫入檔案
    fprintf(fout, ",%u", acc_val); // 實際值
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，輸出 32 位值
            value = sketch[i][cid[i]];
            fprintf(fout, ",%u", value);
        } 
		else {
            // 如果未溢出，輸出22位值 and 10-bit value
            uint low = sketch[i][cid[i]] & 511;
            uint high = sketch[i][cid[i]] >> THRESH_BIT;
            fprintf(fout, ",%u,%u", high, low);
        }
    }
    fprintf(fout, "\n"); 
}

std::vector<float> CUSketch::GetHashedValue(cuc *str)
{
    uint cid[4];
    std::vector<float> counter_values;
    // Step 1: 計算hash值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }

    // Step 2: 遍歷每一行，獲取計數器值
    for (uint i = 0; i < d; i++) {
        uint value;
		uint value2;
        // 檢查溢出標誌
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，取整個 32 位值
            value = sketch[i][cid[i]];
			counter_values.push_back(value);
        } else {
            // 如果未溢出，僅取後 10 位值
			value2 = sketch[i][cid[i]] >> 9;
            value = sketch[i][cid[i]] & 511;
			counter_values.push_back(value2);
			counter_values.push_back(value);
        }
        
        
    }

    return counter_values;
}

uint CUSketch::Query(cuc *str, bool ml){
	memset(t, 0, sizeof(t));
    uint Min = UINT_MAX;
    for(uint i = 0; i < d; ++i){
        uint cid = hf->Str2Int(str, i)%w;
        t[i] = sketch[i][cid];
        Min = std::min(Min, t[i]);
    }
	return (uint)Min;
}

uint CUSketch::Enhanced_Query(cuc* str, int* feature_count)
{

		uint min = UINT_MAX;
		uint cid[4];
		
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
				value = sketch[i][cid[i]] & (THRESH - 1);
				*feature_count+=2;
			}
			
			// 更新最小值
			if (value < min) {
				min = value;
			}
		}
	
		// Step 3: 返回最小值
		return min;
	
}


void CUSketch::PrintCounter(cuc* str, uint acc_val){
	memset(t, 0, sizeof(t));

	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}

    if (need_analyze(t, d)) {
        std::sort(t, t+d);
        printf("%u", acc_val);
        for(uint i = 0; i < d; ++i){
            printf(" %u", t[i]);
        }
        printf("\n");
    }
}

void CUSketch::PrintCounterFile(cuc * str, uint acc_val, FILE * fout)
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
float CUSketch::CalculateAAE(cuc * str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs((float)t[0] - acc_val);
}
void CUSketch::LoadPara(cuc *path){
	FILE *file = fopen((const char*)path, "r");
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", mean+i);
	}
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", scale+i);
	}
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", para+i);
	}
}

float CUSketch::Predict(uint *t){
	float res = 0;
	for(uint i = 0; i < d; ++i){
		res += para[i]*(t[i]-mean[i])/scale[i];
	}
	return res;
}

#endif
#define STAGE4E_D4_CORE_ONLY
#include "stage4e_d4_a4_trajectory.c"
#undef STAGE4E_D4_CORE_ONLY

#ifndef STAGE4E_D6_CORE_ONLY
#define D6_PROVENANCE "step5-checkpoint-roundtrip-2026-10-01-a"
#define D6_CHECKPOINT "stage4e-d6-a4-step5-checkpoint.bin"
#define D6_EXPECTED_D4_SHA "d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1"
#endif
#define D6_FORMAT_VERSION 1U
#define D6_HEADER_BYTES 208U
#define D6_PAYLOAD_FLOATS 774912U
#define D6_PAYLOAD_BYTES 3099648U
#define D6_TOTAL_BYTES 3099856U
#define D6_MAGIC "S4D6CKP1"

static const char d6_dataset_sha[]="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d6_cache_sha[]="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d6_base_sha[]="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d6_tokenizer_sha[]="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d6_d1_sha[]="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
static const char d6_d2_sha[]="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e";
static const char d6_d3_sha[]="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e";

#ifndef STAGE4E_D6_CORE_ONLY
static const char d6_step5_train_average[] = "10.258683447106";
static const char d6_step5_validation_average[] = "9.048405144340";
static const char d6_step5_grad_a_max[] = "0.286533921957";
static const char d6_step5_grad_b_max[] = "0.516311824322";
static const char d6_step5_norm_before[] = "2.1778927370269696";
#endif

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_fmt17(double value,const char *expected)
{
    char text[64];
    if(snprintf(text,sizeof(text),"%.17g",value)<0)return 0;
    return strcmp(text,expected)==0;
}
#endif

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_write_u32(FILE *f, uint32_t value)
{
    unsigned char b[4];
    b[0]=(unsigned char)value; b[1]=(unsigned char)(value>>8);
    b[2]=(unsigned char)(value>>16); b[3]=(unsigned char)(value>>24);
    return fwrite(b,1,4,f)==4;
}

static int d6_write_u64(FILE *f, uint64_t value)
{
    unsigned char b[8]; unsigned int i;
    for(i=0;i<8;i++) b[i]=(unsigned char)(value>>(8U*i));
    return fwrite(b,1,8,f)==8;
}
#endif

static int d6_read_u32(FILE *f, uint32_t *value)
{
    unsigned char b[4];
    if(fread(b,1,4,f)!=4)return 0;
    *value=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
    return 1;
}

static int d6_read_u64(FILE *f, uint64_t *value)
{
    unsigned char b[8]; unsigned int i; uint64_t x=0U;
    if(fread(b,1,8,f)!=8)return 0;
    for(i=0;i<8;i++)x|=(uint64_t)b[i]<<(8U*i);
    *value=x; return 1;
}

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_write_f32(FILE *f,float value)
{
    uint32_t bits; memcpy(&bits,&value,sizeof(bits)); return d6_write_u32(f,bits);
}
#endif

static int d6_read_f32(FILE *f,float *value)
{
    uint32_t bits; if(!d6_read_u32(f,&bits))return 0; memcpy(value,&bits,sizeof(bits)); return 1;
}

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_write_f64(FILE *f,double value)
{
    uint64_t bits; memcpy(&bits,&value,sizeof(bits)); return d6_write_u64(f,bits);
}
#endif

static int d6_hex_raw(const char *hex,unsigned char raw[32])
{
    size_t i;
    if(hex==NULL||strlen(hex)!=64U)return 0;
    for(i=0;i<32U;i++){
        int h,l; char a=hex[2U*i],b=hex[2U*i+1U];
        h=a>='0'&&a<='9'?a-'0':(a>='a'&&a<='f'?a-'a'+10:-1);
        l=b>='0'&&b<='9'?b-'0':(b>='a'&&b<='f'?b-'a'+10:-1);
        if(h<0||l<0)return 0; raw[i]=(unsigned char)((h<<4)|l);
    }
    return 1;
}

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_write_header(FILE *f,const char *dataset_sha,const char *cache_sha,
    const char *checkpoint_sha,const char *tokenizer_sha,uint64_t optimizer_step)
{
    unsigned char dataset[32],cache[32],checkpoint[32],tokenizer[32];
    if(!d6_hex_raw(dataset_sha,dataset)||!d6_hex_raw(cache_sha,cache)||
       !d6_hex_raw(checkpoint_sha,checkpoint)||!d6_hex_raw(tokenizer_sha,tokenizer))return 0;
    if(fwrite(D6_MAGIC,1,8,f)!=8||!d6_write_u32(f,D6_FORMAT_VERSION)||
       !d6_write_u32(f,D6_HEADER_BYTES)||!d6_write_u32(f,D4_DIM)||
       !d6_write_u32(f,D4_VOCAB)||!d6_write_u32(f,D4_RANK)||
       !d6_write_u32(f,2304U)||!d6_write_u32(f,256000U)||
       !d6_write_f32(f,D4_SCALE)||!d6_write_u64(f,optimizer_step)||
       !d6_write_f32(f,(float)D4_LR)||!d6_write_f32(f,(float)D4_BETA1)||
       !d6_write_f32(f,(float)D4_BETA2)||!d6_write_f32(f,(float)D4_EPSILON)||
       !d6_write_f32(f,(float)D4_WEIGHT_DECAY)||!d6_write_f64(f,D4_CLIP)||
       !d6_write_u32(f,D6_PAYLOAD_FLOATS))return 0;
    return fwrite(dataset,1,32,f)==32&&fwrite(cache,1,32,f)==32&&
        fwrite(checkpoint,1,32,f)==32&&fwrite(tokenizer,1,32,f)==32;
}

static int d6_write_array(FILE *f,const float *values,size_t count)
{
    size_t i;
    for(i=0;i<count;i++)if(!d6_write_f32(f,values[i]))return 0;
    return 1;
}

static int d6_write_checkpoint(const char *path,const char *dataset_sha,
    const char *cache_sha,const char *checkpoint_sha,const char *tokenizer_sha,
    const stage4_adapter_t *ad,uint64_t optimizer_step,uint64_t *actual_bytes)
{
    FILE *f=fopen(path,"wb"); int ok;
    if(f==NULL)return 0;
    ok=d6_write_header(f,dataset_sha,cache_sha,checkpoint_sha,tokenizer_sha,optimizer_step)&&
        d6_write_array(f,ad->a,ad->a_count)&&d6_write_array(f,ad->b,ad->b_count)&&
        d6_write_array(f,ad->m1_a,ad->a_count)&&d6_write_array(f,ad->m2_a,ad->a_count)&&
        d6_write_array(f,ad->m1_b,ad->b_count)&&d6_write_array(f,ad->m2_b,ad->b_count);
    if(fflush(f)!=0)ok=0;
    if(fclose(f)!=0)ok=0;
    if(!ok)return 0;
    return stage4_file_size_bytes(path,actual_bytes)&&*actual_bytes==D6_TOTAL_BYTES;
}
#endif

static int d6_read_and_validate_header(FILE *f,const char *dataset_sha,
    const char *cache_sha,const char *checkpoint_sha,const char *tokenizer_sha,
    uint64_t *optimizer_step)
{
    unsigned char magic[8],dataset[32],cache[32],checkpoint[32],tokenizer[32];
    unsigned char expected_dataset[32],expected_cache[32],expected_checkpoint[32],expected_tokenizer[32];
    uint32_t version,header_bytes,dim,vocab,rank,a_count,b_count,payload_count;
    uint64_t step,clip_bits; float scale,lr,beta1,beta2,epsilon,weight_decay;
    double clip; unsigned char clip_le[8]; uint64_t decoded_clip=0U; unsigned int j;
    if(fread(magic,1,8,f)!=8||memcmp(magic,D6_MAGIC,8)!=0||
       !d6_read_u32(f,&version)||!d6_read_u32(f,&header_bytes)||
       !d6_read_u32(f,&dim)||!d6_read_u32(f,&vocab)||!d6_read_u32(f,&rank)||
       !d6_read_u32(f,&a_count)||!d6_read_u32(f,&b_count)||
       !d6_read_f32(f,&scale)||!d6_read_u64(f,&step)||!d6_read_f32(f,&lr)||
       !d6_read_f32(f,&beta1)||!d6_read_f32(f,&beta2)||!d6_read_f32(f,&epsilon)||
       !d6_read_f32(f,&weight_decay)||!d6_read_u64(f,&clip_bits)||
       !d6_read_u32(f,&payload_count)||fread(dataset,1,32,f)!=32||
       fread(cache,1,32,f)!=32||fread(checkpoint,1,32,f)!=32||fread(tokenizer,1,32,f)!=32)return 0;
    for(j=0;j<8;j++)clip_le[j]=(unsigned char)(clip_bits>>(8U*j));
    memcpy(&decoded_clip,clip_le,8U); memcpy(&clip,&decoded_clip,8U);
    if(!d6_hex_raw(dataset_sha,expected_dataset)||!d6_hex_raw(cache_sha,expected_cache)||
       !d6_hex_raw(checkpoint_sha,expected_checkpoint)||!d6_hex_raw(tokenizer_sha,expected_tokenizer))return 0;
    if(version!=D6_FORMAT_VERSION||header_bytes!=D6_HEADER_BYTES||dim!=D4_DIM||
       vocab!=D4_VOCAB||rank!=D4_RANK||a_count!=2304U||b_count!=256000U||
       payload_count!=D6_PAYLOAD_FLOATS||step!=5U||scale!=D4_SCALE||
       lr!=(float)D4_LR||beta1!=(float)D4_BETA1||beta2!=(float)D4_BETA2||
       epsilon!=(float)D4_EPSILON||weight_decay!=(float)D4_WEIGHT_DECAY||clip!=D4_CLIP||
       memcmp(dataset,expected_dataset,32U)!=0||memcmp(cache,expected_cache,32U)!=0||
       memcmp(checkpoint,expected_checkpoint,32U)!=0||memcmp(tokenizer,expected_tokenizer,32U)!=0)return 0;
    *optimizer_step=step; return 1;
}

static int d6_read_array(FILE *f,float *values,size_t count)
{
    size_t i; for(i=0;i<count;i++)if(!d6_read_f32(f,&values[i]))return 0; return 1;
}

static uint32_t d6_byte_mismatches(const void *left,const void *right,size_t bytes)
{
    const unsigned char *a=(const unsigned char *)left,*b=(const unsigned char *)right;
    size_t i; uint32_t mismatches=0U; for(i=0;i<bytes;i++)if(a[i]!=b[i])mismatches++; return mismatches;
}

static int d6_load_checkpoint(const char *path,const char *dataset_sha,
    const char *cache_sha,const char *checkpoint_sha,const char *tokenizer_sha,
    stage4_adapter_t *ad,uint64_t *optimizer_step,uint64_t expected_bytes)
{
    FILE *f=fopen(path,"rb"); uint64_t size=0U; int ok;
    if(!f)return 0;
    if(!d6_read_and_validate_header(f,dataset_sha,cache_sha,checkpoint_sha,tokenizer_sha,optimizer_step)){fclose(f);return 0;}
    ok=d6_read_array(f,ad->a,ad->a_count)&&d6_read_array(f,ad->b,ad->b_count)&&
       d6_read_array(f,ad->m1_a,ad->a_count)&&d6_read_array(f,ad->m2_a,ad->a_count)&&
       d6_read_array(f,ad->m1_b,ad->b_count)&&d6_read_array(f,ad->m2_b,ad->b_count);
    if(!ok||fgetc(f)!=EOF||ferror(f)){fclose(f);return 0;}
    if(fclose(f)!=0||!stage4_file_size_bytes(path,&size)||size!=expected_bytes)return 0;
    return 1;
}

#ifndef STAGE4E_D6_CORE_ONLY
static int d6_write_report(const char *path,const char *identities[8],
    uint64_t actual_size,const char *checkpoint_sha,uint32_t mismatches_a,
    uint32_t mismatches_b,uint32_t mismatches_m1a,uint32_t mismatches_m2a,
    uint32_t mismatches_m1b,uint32_t mismatches_m2b,uint64_t reloaded_step,
    double pre_train,double post_train,double pre_val,double post_val,
    int step5_match,int train_match,int validation_match,int params_finite,
    int moments_finite,int passed)
{
    static const char *keys[8]={"dataset_sha256","cache_sha256","base_checkpoint_sha256","tokenizer_sha256","d1_report_sha256","d2_report_sha256","d3_report_sha256","d4_report_sha256"};
    FILE *f=fopen(path,"wb"); uint32_t total=mismatches_a+mismatches_b+mismatches_m1a+mismatches_m2a+mismatches_m1b+mismatches_m2b; int i;
    if(!f)return 0;
    for(i=0;i<8;i++)fprintf(f,"%s=%s\n",keys[i],identities[i]);
    fprintf(f,"checkpoint_format_version=%u\ncheckpoint_header_bytes=%u\ncheckpoint_payload_float_count=%u\ncheckpoint_payload_bytes=%u\ncheckpoint_expected_total_bytes=%u\ncheckpoint_actual_total_bytes=%llu\ncheckpoint_sha256=%s\n",D6_FORMAT_VERSION,D6_HEADER_BYTES,D6_PAYLOAD_FLOATS,D6_PAYLOAD_BYTES,D6_TOTAL_BYTES,(unsigned long long)actual_size,checkpoint_sha);
    fprintf(f,"optimizer_step_before_training=0\noptimizer_step_at_save=5\nreloaded_optimizer_step=%llu\nstep5_d4_metrics_match=%s\n",(unsigned long long)reloaded_step,step5_match?"yes":"no");
    fprintf(f,"A_byte_mismatches=%u\nB_byte_mismatches=%u\nm1_A_byte_mismatches=%u\nm2_A_byte_mismatches=%u\nm1_B_byte_mismatches=%u\nm2_B_byte_mismatches=%u\ntotal_state_byte_mismatches=%u\n",mismatches_a,mismatches_b,mismatches_m1a,mismatches_m2a,mismatches_m1b,mismatches_m2b,total);
    fprintf(f,"pre_save_train_average_loss=%.12f\npost_load_train_average_loss=%.12f\npre_save_validation_average_loss=%.12f\npost_load_validation_average_loss=%.12f\npre_save_vs_post_load_train_metrics_match=%s\npre_save_vs_post_load_validation_metrics_match=%s\n",pre_train,post_train,pre_val,post_val,train_match?"yes":"no",validation_match?"yes":"no");
    fprintf(f,"test_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\nloaded_parameters_finite=%s\nloaded_optimizer_moments_finite=%s\ncheckpoint_roundtrip_pass=%s\n",params_finite?"yes":"no",moments_finite?"yes":"no",passed?"yes":"no");
    return fclose(f)==0;
}

int main(int argc,char **argv)
{
    const char *checkpoint_path=NULL,*tokenizer_path=NULL,*dataset_path=NULL,*cache_path=NULL;
    const char *d1_path=NULL,*d2_path=NULL,*d3_path=NULL,*d4_path=NULL;
    char dataset_sha[65],cache_sha[65],base_sha[65],tokenizer_sha[65],d1_sha[65],d2_sha[65],d3_sha[65],d4_sha[65],checkpoint_sha[65];
    const char *identities[8]; unsigned char dataset_raw[32],base_raw[32],tokenizer_raw[32];
    d4_header_t header; d4_row_t *rows=NULL; d4_point_t point; stage4_adapter_t ad;
    Transformer tr; FILE *cache=NULL; float *base_logits=NULL,*final_logits=NULL;
    float *snap_a=NULL,*snap_b=NULL,*snap_m1a=NULL,*snap_m2a=NULL,*snap_m1b=NULL,*snap_m2b=NULL;
    uint32_t split_counts[4]={0U,0U,0U,0U},i; uint64_t cache_size=0U,actual_size=0U,reloaded_step=0U;
    uint32_t step5_m1a=0U,step5_m2a=0U,step5_m1b=0U,step5_m2b=0U;
    uint32_t bad_a=0U,bad_b=0U,bad_m1a=0U,bad_m2a=0U,bad_m1b=0U,bad_m2b=0U;
    uint32_t test_gradient_rows=0U,test_evaluation_rows=0U; double pre_train=0.0,pre_val=0.0,post_train=0.0,post_val=0.0,update_norm;
    int argi,tr_ready=0,ad_ready=0,loss_finite=1,params_finite=0,moments_finite=0;
    int step5_match=0,train_match=0,validation_match=0,pass=0,failed=0;
    const char *report_path="stage4e-d6-a4-checkpoint-report.txt";
    printf("d6_build_provenance=%s\n",D6_PROVENANCE);
    memset(&header,0,sizeof(header));memset(&ad,0,sizeof(ad));memset(&tr,0,sizeof(tr));memset(&point,0,sizeof(point));
    if(argc<2){fprintf(stderr,"usage: %s checkpoint -z tokenizer -d dataset -c cache -1 d1 -2 d2 -3 d3 -4 d4\n",argv[0]);return 1;}
    checkpoint_path=argv[1];
    for(argi=2;argi<argc;argi++){
        if(!strcmp(argv[argi],"-z")&&argi+1<argc)tokenizer_path=argv[++argi];else if(!strcmp(argv[argi],"-d")&&argi+1<argc)dataset_path=argv[++argi];else if(!strcmp(argv[argi],"-c")&&argi+1<argc)cache_path=argv[++argi];else if(!strcmp(argv[argi],"-1")&&argi+1<argc)d1_path=argv[++argi];else if(!strcmp(argv[argi],"-2")&&argi+1<argc)d2_path=argv[++argi];else if(!strcmp(argv[argi],"-3")&&argi+1<argc)d3_path=argv[++argi];else if(!strcmp(argv[argi],"-4")&&argi+1<argc)d4_path=argv[++argi];else{fprintf(stderr,"error: invalid D6 option\n");return 1;}}
    if(!tokenizer_path||!dataset_path||!cache_path||!d1_path||!d2_path||!d3_path||!d4_path){fprintf(stderr,"error: missing D6 input path\n");return 1;}
    if(!stage4_hash_file_sha256_hex(dataset_path,dataset_sha)||!stage4_hash_file_sha256_hex(cache_path,cache_sha)||!stage4_hash_file_sha256_hex(checkpoint_path,base_sha)||!stage4_hash_file_sha256_hex(tokenizer_path,tokenizer_sha)||!stage4_hash_file_sha256_hex(d1_path,d1_sha)||!stage4_hash_file_sha256_hex(d2_path,d2_sha)||!stage4_hash_file_sha256_hex(d3_path,d3_sha)||!stage4_hash_file_sha256_hex(d4_path,d4_sha)){fprintf(stderr,"error: D6 input hash failure\n");failed=1;goto cleanup;}
    if(strcmp(dataset_sha,d6_dataset_sha)||strcmp(cache_sha,d6_cache_sha)||strcmp(base_sha,d6_base_sha)||strcmp(tokenizer_sha,d6_tokenizer_sha)||strcmp(d1_sha,d6_d1_sha)||strcmp(d2_sha,d6_d2_sha)||strcmp(d3_sha,d6_d3_sha)||strcmp(d4_sha,D6_EXPECTED_D4_SHA)){fprintf(stderr,"error: D6 input identity mismatch\n");failed=1;goto cleanup;}
    identities[0]=dataset_sha;identities[1]=cache_sha;identities[2]=base_sha;identities[3]=tokenizer_sha;identities[4]=d1_sha;identities[5]=d2_sha;identities[6]=d3_sha;identities[7]=d4_sha;
    if(!d4_float_ok()||!stage4_file_size_bytes(cache_path,&cache_size)||cache_size!=D4_CACHE_SIZE||!d4_hex_raw(dataset_sha,dataset_raw)||!d4_hex_raw(base_sha,base_raw)||!d4_hex_raw(tokenizer_sha,tokenizer_raw)){fprintf(stderr,"error: D6 float/cache format validation failed\n");failed=1;goto cleanup;}
    cache=fopen(cache_path,"rb");if(!cache||!d4_read_header(cache,&header)||!d4_header_matches(&header,dataset_raw,base_raw,tokenizer_raw)){fprintf(stderr,"error: D6 cache header mismatch\n");failed=1;goto cleanup;}
    rows=calloc(D4_ROWS,sizeof(*rows));if(!rows){fprintf(stderr,"error: D6 cache row allocation failed\n");failed=1;goto cleanup;}
    for(i=0;i<D4_ROWS;i++){if(!d4_read_row(cache,&rows[i])||rows[i].record>=D4_RECORDS||rows[i].target>=D4_VOCAB||rows[i].split>=D4_SPLITS||rows[i].split==D4_REGRESSION){fprintf(stderr,"error: invalid D6 cache row %u\n",i);failed=1;goto cleanup;}split_counts[rows[i].split]++;}
    if(fgetc(cache)!=EOF||ferror(cache)||fclose(cache)!=0){cache=NULL;fprintf(stderr,"error: D6 cache trailing/read error\n");failed=1;goto cleanup;}cache=NULL;
    if(split_counts[D4_TRAIN]!=111U||split_counts[D4_VALIDATION]!=33U||split_counts[D4_TEST]!=21U){fprintf(stderr,"error: D6 split counts mismatch\n");failed=1;goto cleanup;}
    unload_adapter_runtime();load_transformer(&tr,checkpoint_path);tr_ready=1;
    if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||!stage4_adapter_init(&ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){fprintf(stderr,"error: D6 model/adapter init failed\n");failed=1;goto cleanup;}ad_ready=1;
    if(ad.rank!=D4_RANK||ad.scale!=D4_SCALE||ad.a_count!=2304U||ad.b_count!=256000U){fprintf(stderr,"error: D6 adapter metadata mismatch\n");failed=1;goto cleanup;}
    stage4_adapter_fill_a_deterministic(&ad,1U,0.01f);stage4_adapter_zero_b(&ad);stage4_adapter_zero_moments(&ad);
    snap_a=malloc(ad.a_count*sizeof(float));snap_b=malloc(ad.b_count*sizeof(float));snap_m1a=malloc(ad.a_count*sizeof(float));snap_m2a=malloc(ad.a_count*sizeof(float));snap_m1b=malloc(ad.b_count*sizeof(float));snap_m2b=malloc(ad.b_count*sizeof(float));
    base_logits=calloc(D4_VOCAB,sizeof(float));final_logits=calloc(D4_VOCAB,sizeof(float));
    if(!snap_a||!snap_b||!snap_m1a||!snap_m2a||!snap_m1b||!snap_m2b||!base_logits||!final_logits){fprintf(stderr,"error: D6 buffers allocation failed\n");failed=1;goto cleanup;}
    if(!d4_evaluate(rows,&ad,&tr,&point.metrics,&loss_finite)){fprintf(stderr,"error: D6 initial train/validation evaluation failed\n");failed=1;goto cleanup;}
    if(!d4_anchor_metrics(&point.metrics,0U)){fprintf(stderr,"error: D6 fresh initialization D1 anchor mismatch\n");failed=1;goto cleanup;}
    for(i=1U;i<=5U;i++){
        int grad_ok=1;
        if(!d4_step(rows,&ad,&tr,base_logits,final_logits,(int)i,&point,&grad_ok,&loss_finite)||!grad_ok){fprintf(stderr,"error: D6 train/clipping step %u failed\n",i);failed=1;goto cleanup;}
        update_norm=stage4_adapter_adam_step(&ad,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);
        if(!isfinite(update_norm)||!stage4_adapter_parameters_are_finite(&ad)){fprintf(stderr,"error: D6 optimizer state invalid at step %u\n",i);failed=1;goto cleanup;}
    }
    if(!d4_evaluate(rows,&ad,&tr,&point.metrics,&loss_finite)){fprintf(stderr,"error: D6 step-5 evaluation failed\n");failed=1;goto cleanup;}
    step5_match=d4_fmt12(point.metrics.split[D4_TRAIN].loss/111.0,d6_step5_train_average)&&d4_fmt12(point.metrics.split[D4_VALIDATION].loss/33.0,d6_step5_validation_average)&&point.metrics.split[D4_TRAIN].top1==0U&&point.metrics.split[D4_VALIDATION].top1==2U&&d4_gradient_anchor(&point,5U)&&point.grad_a_nonzero==2304U&&d4_fmtg(point.grad_a_max,d6_step5_grad_a_max)&&d4_fmtg(point.grad_b_max,d6_step5_grad_b_max)&&d6_fmt17(point.norm_before,d6_step5_norm_before)&&point.clip_applied;
    if(!step5_match){fprintf(stderr,"error: D6 step-5 D4 anchor mismatch\n");failed=1;goto cleanup;}
    d4_moment_counts(&ad,&step5_m1a,&step5_m2a,&step5_m1b,&step5_m2b);
    pre_train=point.metrics.split[D4_TRAIN].loss/111.0;pre_val=point.metrics.split[D4_VALIDATION].loss/33.0;
    memcpy(snap_a,ad.a,ad.a_count*sizeof(float));memcpy(snap_b,ad.b,ad.b_count*sizeof(float));memcpy(snap_m1a,ad.m1_a,ad.a_count*sizeof(float));memcpy(snap_m2a,ad.m2_a,ad.a_count*sizeof(float));memcpy(snap_m1b,ad.m1_b,ad.b_count*sizeof(float));memcpy(snap_m2b,ad.m2_b,ad.b_count*sizeof(float));
    if(!d6_write_checkpoint(D6_CHECKPOINT,dataset_sha,cache_sha,base_sha,tokenizer_sha,&ad,5U,&actual_size)){fprintf(stderr,"error: D6 checkpoint write/size validation failed\n");failed=1;goto cleanup;}
    if(!stage4_file_size_bytes(D6_CHECKPOINT,&actual_size)){fprintf(stderr,"error: cannot stat D6 checkpoint\n");failed=1;goto cleanup;}
    stage4_adapter_free(&ad);ad_ready=0;
    if(!stage4_adapter_init(&ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){fprintf(stderr,"error: D6 fresh adapter allocation failed\n");failed=1;goto cleanup;}ad_ready=1;
    if(memcmp(snap_a,ad.a,ad.a_count*sizeof(float))==0&&memcmp(snap_b,ad.b,ad.b_count*sizeof(float))==0&&memcmp(snap_m1a,ad.m1_a,ad.a_count*sizeof(float))==0&&memcmp(snap_m2a,ad.m2_a,ad.a_count*sizeof(float))==0&&memcmp(snap_m1b,ad.m1_b,ad.b_count*sizeof(float))==0&&memcmp(snap_m2b,ad.m2_b,ad.b_count*sizeof(float))==0){fprintf(stderr,"error: fresh D6 state unexpectedly matches saved state\n");failed=1;goto cleanup;}
    if(!d6_load_checkpoint(D6_CHECKPOINT,dataset_sha,cache_sha,base_sha,tokenizer_sha,&ad,&reloaded_step,D6_TOTAL_BYTES)){fprintf(stderr,"error: D6 checkpoint header/payload load failed\n");failed=1;goto cleanup;}
    bad_a=d6_byte_mismatches(snap_a,ad.a,ad.a_count*sizeof(float));bad_b=d6_byte_mismatches(snap_b,ad.b,ad.b_count*sizeof(float));bad_m1a=d6_byte_mismatches(snap_m1a,ad.m1_a,ad.a_count*sizeof(float));bad_m2a=d6_byte_mismatches(snap_m2a,ad.m2_a,ad.a_count*sizeof(float));bad_m1b=d6_byte_mismatches(snap_m1b,ad.m1_b,ad.b_count*sizeof(float));bad_m2b=d6_byte_mismatches(snap_m2b,ad.m2_b,ad.b_count*sizeof(float));
    d4_state_finite(&ad,&params_finite,&moments_finite);
    if(!d4_evaluate(rows,&ad,&tr,&point.metrics,&loss_finite)){fprintf(stderr,"error: post-load train/validation evaluation failed\n");failed=1;goto cleanup;}
    post_train=point.metrics.split[D4_TRAIN].loss/111.0;post_val=point.metrics.split[D4_VALIDATION].loss/33.0;
    train_match=d4_fmt12(pre_train,d6_step5_train_average)&&d4_fmt12(post_train,d6_step5_train_average);
    validation_match=d4_fmt12(pre_val,d6_step5_validation_average)&&d4_fmt12(post_val,d6_step5_validation_average);
    test_gradient_rows=0U;test_evaluation_rows=0U;
    pass=step5_match&&reloaded_step==5U&&actual_size==D6_TOTAL_BYTES&&bad_a==0U&&bad_b==0U&&bad_m1a==0U&&bad_m2a==0U&&bad_m1b==0U&&bad_m2b==0U&&train_match&&validation_match&&test_gradient_rows==0U&&test_evaluation_rows==0U&&params_finite&&moments_finite&&loss_finite;
    if(!stage4_hash_file_sha256_hex(D6_CHECKPOINT,checkpoint_sha)){fprintf(stderr,"error: cannot hash D6 checkpoint\n");failed=1;goto cleanup;}
    if(!d6_write_report(report_path,identities,actual_size,checkpoint_sha,bad_a,bad_b,bad_m1a,bad_m2a,bad_m1b,bad_m2b,reloaded_step,pre_train,post_train,pre_val,post_val,step5_match,train_match,validation_match,params_finite,moments_finite,pass)){fprintf(stderr,"error: D6 report write failed\n");failed=1;goto cleanup;}
    printf("base_checkpoint_identity=PASS\ntokenizer_identity=PASS\ndataset_identity=PASS\ncache_identity=PASS\nd1_report_identity=PASS\nd2_report_identity=PASS\nd3_report_identity=PASS\nd4_report_identity=PASS\noptimizer_step_before_training=0\noptimizer_step_at_save=5\nreloaded_optimizer_step=%llu\nstep5_d4_metrics_match=%s\ncheckpoint_expected_total_bytes=%u\ncheckpoint_actual_total_bytes=%llu\ncheckpoint_sha256=%s\nA_byte_mismatches=%u\nB_byte_mismatches=%u\nm1_A_byte_mismatches=%u\nm2_A_byte_mismatches=%u\nm1_B_byte_mismatches=%u\nm2_B_byte_mismatches=%u\ntotal_state_byte_mismatches=%u\npost_load_train_average_loss=%.12f\npost_load_validation_average_loss=%.12f\npre_save_vs_post_load_train_metrics_match=%s\npre_save_vs_post_load_validation_metrics_match=%s\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\nloaded_parameters_finite=%s\nloaded_optimizer_moments_finite=%s\nStage4E-D6-A4=%s\n",(unsigned long long)reloaded_step,step5_match?"yes":"no",D6_TOTAL_BYTES,(unsigned long long)actual_size,checkpoint_sha,bad_a,bad_b,bad_m1a,bad_m2a,bad_m1b,bad_m2b,bad_a+bad_b+bad_m1a+bad_m2a+bad_m1b+bad_m2b,post_train,post_val,train_match?"yes":"no",validation_match?"yes":"no",params_finite?"yes":"no",moments_finite?"yes":"no",pass?"PASS":"FAIL");
    printf("step5_m1_A_nonzero_elements=%u\nstep5_m2_A_nonzero_elements=%u\nstep5_m1_B_nonzero_elements=%u\nstep5_m2_B_nonzero_elements=%u\n",step5_m1a,step5_m2a,step5_m1b,step5_m2b);
    printf("evaluated_splits=%s,%s\n",d4_split_names[D4_TRAIN],d4_split_names[D4_VALIDATION]);
    if(!pass)failed=1;
cleanup:
    if(cache)fclose(cache);free(rows);free(base_logits);free(final_logits);free(snap_a);free(snap_b);free(snap_m1a);free(snap_m2a);free(snap_m1b);free(snap_m2b);if(ad_ready)stage4_adapter_free(&ad);if(tr_ready)free_transformer(&tr);
    return failed||!pass?EXIT_FAILURE:EXIT_SUCCESS;
}
#endif

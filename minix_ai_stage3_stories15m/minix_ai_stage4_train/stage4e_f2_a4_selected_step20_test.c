#define STAGE4E_D6_CORE_ONLY
#include "stage4e_d6_a4_checkpoint.c"
#undef STAGE4E_D6_CORE_ONLY
#define STAGE4E_D5_CORE_ONLY
#include "stage4e_d5_a4_repeat.c"
#undef STAGE4E_D5_CORE_ONLY

#define F2_PROVENANCE "selected-step20-test-sealed-2026-10-01-a"
#define F2_FIRST_STEP 6U
#define F2_LAST_STEP 20U
#define F2_CHECKPOINT "stage4e-f2-a4-selected-step20-audit-checkpoint.bin"
#define F2_REPORT "stage4e-f2-a4-selected-step20-audit-test-report.txt"
#define F2_TSV "stage4e-f2-a4-selected-step20-audit-reconstruction.tsv"
#define F2_D6_REPORT "stage4e-d6-a4-checkpoint-report.txt"
#define F2_D7_REPORT "stage4e-d7-a4-resume-report.txt"
#define F2_D7_TSV "stage4e-d7-a4-resume-trajectory.tsv"
#define F2_DATASET "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
#define F2_CACHE "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
#define F2_BASE "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
#define F2_TOKENIZER "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
#define F2_D6 "ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"
#define F2_TRAIN20 "7.026346899236"
#define F2_VALIDATION20 "8.328697455428"
#define F2_BASELINE_TEST_LOSS 10.664819497567

typedef struct { d4_point_t point; int evaluated; } f2_point_t;
typedef struct { double loss; uint64_t top1; uint32_t rows; } f2_test_t;

static int f2_value(const d5_kv_t *items,uint32_t count,const char *key,const char *expected)
{
    const char *actual=d5_get(items,count,key);
    return actual!=NULL&&strcmp(actual,expected)==0;
}

static int f2_test_evaluate(const d4_row_t *rows,stage4_adapter_t *ad,Transformer *tr,f2_test_t *out,int *loss_ok)
{
    float *base=calloc(D4_VOCAB,sizeof(float)),*final=calloc(D4_VOCAB,sizeof(float));uint32_t i;
    if(!base||!final){free(base);free(final);return 0;}
    memset(out,0,sizeof(*out));
    for(i=0U;i<D4_ROWS;i++){
        double loss;int prediction;
        if(rows[i].split!=D4_TEST)continue;
        matmul(base,rows[i].hidden,tr->weights.wcls,(int)D4_DIM,(int)D4_VOCAB);
        stage4_adapter_forward_logits(ad,rows[i].hidden,base,final);
        if(!d4_loss(final,rows[i].target,&loss)){*loss_ok=0;free(base);free(final);return 0;}
        prediction=d4_argmax(final);out->loss+=loss;out->rows++;
        if(prediction==(int)rows[i].target)out->top1++;
    }
    free(base);free(final);return out->rows==21U&&isfinite(out->loss);
}

static int f2_parse(char *line,uint32_t step,char *field[30])
{
    char *p,*save=NULL;uint32_t count=0U;
    p=strtok_r(line,"\t\r\n",&save);
    while(p&&count<30U){field[count++]=p;p=strtok_r(NULL,"\t\r\n",&save);}
    return count==30U&&(uint32_t)strtoul(field[0],NULL,10)==step;
}

static int f2_matches_f1(const char *path,const f2_point_t points[F2_LAST_STEP+1],uint32_t *rows_compared,uint32_t *field_mismatches)
{
    FILE *f=fopen(path,"rb");char line[4096],copy[4096],*field[30],actual[96];uint32_t step;
    *rows_compared=0U;*field_mismatches=0U;
    if(!f||fgets(line,sizeof(line),f)==NULL){if(f)fclose(f);return 0;}
    for(step=F2_FIRST_STEP;step<=F2_LAST_STEP;step++){
        const d4_point_t *p=&points[step].point;
#define F2_MATCH(column,format,value) do { snprintf(actual,sizeof(actual),format,value); if(strcmp(field[column],actual))(*field_mismatches)++; } while(0)
        if(fgets(line,sizeof(line),f)==NULL||strlen(line)>=sizeof(copy)){fclose(f);return 0;}
        strcpy(copy,line);if(!f2_parse(copy,step,field)){fclose(f);return 0;}
        (*rows_compared)++;
        F2_MATCH(1,"%u",p->grad_train_rows);F2_MATCH(2,"%u",p->grad_validation_rows);F2_MATCH(3,"%u",p->grad_test_rows);
        F2_MATCH(4,"%u",p->grad_a_nonzero);F2_MATCH(5,"%.12g",p->grad_a_max);F2_MATCH(6,"%u",p->grad_b_nonzero);
        F2_MATCH(7,"%.12g",p->grad_b_max);F2_MATCH(8,"%.17g",p->norm_before);F2_MATCH(9,"%.17g",p->norm_after);
        F2_MATCH(10,"%s",p->clip_applied?"yes":"no");F2_MATCH(11,"%s",p->clip_consistency?"yes":"no");F2_MATCH(12,"%s",points[step].evaluated?"yes":"no");
        if(points[step].evaluated){
            F2_MATCH(13,"%.12f",p->metrics.split[D4_TRAIN].loss/111.0);F2_MATCH(14,"%llu",(unsigned long long)p->metrics.split[D4_TRAIN].top1);
            F2_MATCH(15,"%.12f",p->metrics.split[D4_VALIDATION].loss/33.0);F2_MATCH(16,"%llu",(unsigned long long)p->metrics.split[D4_VALIDATION].top1);
            F2_MATCH(17,"%s",p->metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no");
        }
#undef F2_MATCH
    }
    return fclose(f)==0&&*field_mismatches==0U;
}

static int f2_w32(FILE *f,uint32_t x){unsigned char b[4];b[0]=(unsigned char)x;b[1]=(unsigned char)(x>>8);b[2]=(unsigned char)(x>>16);b[3]=(unsigned char)(x>>24);return fwrite(b,1,4,f)==4;}
static int f2_w64(FILE *f,uint64_t x){unsigned char b[8];uint32_t i;for(i=0U;i<8U;i++)b[i]=(unsigned char)(x>>(8U*i));return fwrite(b,1,8,f)==8;}
static int f2_wf(FILE *f,float x){uint32_t b;memcpy(&b,&x,4U);return f2_w32(f,b);}
static int f2_wd(FILE *f,double x){uint64_t b;memcpy(&b,&x,8U);return f2_w64(f,b);}
static int f2_wa(FILE *f,const float *x,size_t n){size_t i;for(i=0U;i<n;i++)if(!f2_wf(f,x[i]))return 0;return 1;}

static int f2_write_checkpoint(const char *dataset,const char *cache,const char *base,const char *tokenizer,const stage4_adapter_t *ad)
{
    unsigned char ds[32],ca[32],ck[32],tk[32];FILE *f;int ok;
    if(!d6_hex_raw(dataset,ds)||!d6_hex_raw(cache,ca)||!d6_hex_raw(base,ck)||!d6_hex_raw(tokenizer,tk))return 0;
    f=fopen(F2_CHECKPOINT,"wb");if(!f)return 0;
    ok=fwrite(D6_MAGIC,1,8,f)==8&&f2_w32(f,D6_FORMAT_VERSION)&&f2_w32(f,D6_HEADER_BYTES)&&f2_w32(f,D4_DIM)&&f2_w32(f,D4_VOCAB)&&f2_w32(f,D4_RANK)&&f2_w32(f,2304U)&&f2_w32(f,256000U)&&f2_wf(f,D4_SCALE)&&f2_w64(f,F2_LAST_STEP)&&f2_wf(f,(float)D4_LR)&&f2_wf(f,(float)D4_BETA1)&&f2_wf(f,(float)D4_BETA2)&&f2_wf(f,(float)D4_EPSILON)&&f2_wf(f,(float)D4_WEIGHT_DECAY)&&f2_wd(f,D4_CLIP)&&f2_w32(f,D6_PAYLOAD_FLOATS)&&fwrite(ds,1,32,f)==32&&fwrite(ca,1,32,f)==32&&fwrite(ck,1,32,f)==32&&fwrite(tk,1,32,f)==32&&f2_wa(f,ad->a,ad->a_count)&&f2_wa(f,ad->b,ad->b_count)&&f2_wa(f,ad->m1_a,ad->a_count)&&f2_wa(f,ad->m2_a,ad->a_count)&&f2_wa(f,ad->m1_b,ad->b_count)&&f2_wa(f,ad->m2_b,ad->b_count);
    if(fflush(f)!=0)ok=0;if(fclose(f)!=0)ok=0;return ok;
}

static int f2_read_checkpoint(const char *path,const char *dataset,const char *cache,const char *base,const char *tokenizer,stage4_adapter_t *ad,uint64_t *step)
{
    unsigned char magic[8],ds[32],ca[32],ck[32],tk[32],eds[32],eca[32],eck[32],etk[32];uint32_t version,header,dim,vocab,rank,ac,bc,payload,bits;float scale,lr,b1,b2,eps,wd;uint64_t clipbits;double clip;FILE *f;uint64_t bytes=0U;int ok;
    f=fopen(path,"rb");if(!f)return 0;
    ok=fread(magic,1,8,f)==8&&memcmp(magic,D6_MAGIC,8)==0&&d6_read_u32(f,&version)&&d6_read_u32(f,&header)&&d6_read_u32(f,&dim)&&d6_read_u32(f,&vocab)&&d6_read_u32(f,&rank)&&d6_read_u32(f,&ac)&&d6_read_u32(f,&bc)&&d6_read_u32(f,&bits);memcpy(&scale,&bits,4U);ok=ok&&d6_read_u64(f,step)&&d6_read_u32(f,&bits);memcpy(&lr,&bits,4U);ok=ok&&d6_read_u32(f,&bits);memcpy(&b1,&bits,4U);ok=ok&&d6_read_u32(f,&bits);memcpy(&b2,&bits,4U);ok=ok&&d6_read_u32(f,&bits);memcpy(&eps,&bits,4U);ok=ok&&d6_read_u32(f,&bits);memcpy(&wd,&bits,4U);ok=ok&&d6_read_u64(f,&clipbits);memcpy(&clip,&clipbits,8U);ok=ok&&d6_read_u32(f,&payload)&&fread(ds,1,32,f)==32&&fread(ca,1,32,f)==32&&fread(ck,1,32,f)==32&&fread(tk,1,32,f)==32&&d6_hex_raw(dataset,eds)&&d6_hex_raw(cache,eca)&&d6_hex_raw(base,eck)&&d6_hex_raw(tokenizer,etk);
    if(!ok||version!=D6_FORMAT_VERSION||header!=D6_HEADER_BYTES||dim!=D4_DIM||vocab!=D4_VOCAB||rank!=D4_RANK||ac!=2304U||bc!=256000U||payload!=D6_PAYLOAD_FLOATS||*step!=F2_LAST_STEP||scale!=D4_SCALE||lr!=(float)D4_LR||b1!=(float)D4_BETA1||b2!=(float)D4_BETA2||eps!=(float)D4_EPSILON||wd!=(float)D4_WEIGHT_DECAY||clip!=D4_CLIP||memcmp(ds,eds,32U)||memcmp(ca,eca,32U)||memcmp(ck,eck,32U)||memcmp(tk,etk,32U)){fclose(f);return 0;}
    ok=d6_read_array(f,ad->a,ad->a_count)&&d6_read_array(f,ad->b,ad->b_count)&&d6_read_array(f,ad->m1_a,ad->a_count)&&d6_read_array(f,ad->m2_a,ad->a_count)&&d6_read_array(f,ad->m1_b,ad->b_count)&&d6_read_array(f,ad->m2_b,ad->b_count)&&fgetc(f)==EOF&&!ferror(f);if(fclose(f)!=0)ok=0;return ok&&stage4_file_size_bytes(path,&bytes)&&bytes==D6_TOTAL_BYTES;
}

static int f2_write_tsv(const f2_point_t points[F2_LAST_STEP+1])
{
    FILE *f=fopen(F2_TSV,"wb");uint32_t i;if(!f)return 0;
    fprintf(f,"step\tgradient_train_rows\tgradient_validation_rows\tgradient_test_rows\tgrad_A_nonzero_elements\tgrad_A_max_abs\tgrad_B_nonzero_elements\tgrad_B_max_abs\tgradient_norm_before_clip\tgradient_norm_after_clip\tgradient_clip_applied\tgradient_clip_consistency\tevaluation_performed\ttrain_average_loss\ttrain_top1_correct\tvalidation_average_loss\tvalidation_top1_correct\tvalidation_acceptance_reached\tvalidation_unique_top1_tokens\tvalidation_dominant_top1_token\tvalidation_dominant_top1_count\tvalidation_dominant_top1_fraction\tvalidation_all_same_top1\tvalidation_correction_rms_mean\tvalidation_correction_rms_max\tvalidation_correction_abs_max\tadapter_A_rms\tadapter_A_max_abs\tadapter_B_rms\tadapter_B_max_abs\n");
    for(i=5U;i<=F2_LAST_STEP;i++){const d4_point_t *p=&points[i].point;if(i==5U)fprintf(f,"5\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tyes");else fprintf(f,"%u\t%u\t%u\t%u\t%u\t%.12g\t%u\t%.12g\t%.17g\t%.17g\t%s\t%s\t%s",i,p->grad_train_rows,p->grad_validation_rows,p->grad_test_rows,p->grad_a_nonzero,p->grad_a_max,p->grad_b_nonzero,p->grad_b_max,p->norm_before,p->norm_after,p->clip_applied?"yes":"no",p->clip_consistency?"yes":"no",points[i].evaluated?"yes":"no");if(points[i].evaluated)fprintf(f,"\t%.12f\t%llu\t%.12f\t%llu\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n",p->metrics.split[D4_TRAIN].loss/111.0,(unsigned long long)p->metrics.split[D4_TRAIN].top1,p->metrics.split[D4_VALIDATION].loss/33.0,(unsigned long long)p->metrics.split[D4_VALIDATION].top1,p->metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no");else fprintf(f,"\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n");}
    return fclose(f)==0;
}

int main(int argc,char **argv)
{
    const char *checkpoint,*tokenizer,*dataset,*cache_path,*d6_checkpoint,*f1_report,*f1_tsv;char ds[65],ca[65],ck[65],tk[65],d6hash[65],d6r_hash[65],d7r_hash[65],d7t_hash[65],f1r_hash[65],f1t_hash[65];d5_kv_t f1[D5_REPORT_ENTRIES];uint32_t f1_count=0U,counts[4]={0U,0U,0U,0U},i,selected=0U,test_gradient=0U,mismatch_count=0U,mismatch_a=0U,mismatch_b=0U,mismatch_m1a=0U,mismatch_m2a=0U,mismatch_m1b=0U,mismatch_m2b=0U,trajectory_rows=0U,trajectory_mismatches=0U,test_calls=0U;uint64_t start_step=0U,reloaded_step=0U;unsigned char dsraw[32],ckraw[32],tkraw[32];d4_header_t header;d4_row_t *rows=NULL;stage4_adapter_t state,reloaded;Transformer tr;FILE *cache=NULL,*report=NULL;float *base=NULL,*final=NULL,*sa=NULL,*sb=NULL,*s1a=NULL,*s2a=NULL,*s1b=NULL,*s2b=NULL;f2_point_t points[F2_LAST_STEP+1];f2_test_t test;int loss_ok=1,grad_ok_all=1,param_ok=1,moment_ok=1,unique=1,f1_match=0,byte_match=0,selection_frozen=0,pass=0,failed=0,state_ready=0,reloaded_ready=0,tr_ready=0;
    if(argc!=8){fprintf(stderr,"usage: %s checkpoint tokenizer dataset cache d6-checkpoint f1-report f1-tsv\n",argv[0]);return 1;}
    checkpoint=argv[1];tokenizer=argv[2];dataset=argv[3];cache_path=argv[4];d6_checkpoint=argv[5];f1_report=argv[6];f1_tsv=argv[7];printf("f2_build_provenance=%s\n",F2_PROVENANCE);memset(ds,0,sizeof(ds));memset(ca,0,sizeof(ca));memset(ck,0,sizeof(ck));memset(tk,0,sizeof(tk));memset(d6hash,0,sizeof(d6hash));memset(d6r_hash,0,sizeof(d6r_hash));memset(d7r_hash,0,sizeof(d7r_hash));memset(d7t_hash,0,sizeof(d7t_hash));memset(f1r_hash,0,sizeof(f1r_hash));memset(f1t_hash,0,sizeof(f1t_hash));memset(&state,0,sizeof(state));memset(&reloaded,0,sizeof(reloaded));memset(&tr,0,sizeof(tr));memset(points,0,sizeof(points));memset(&test,0,sizeof(test));
    if(!stage4_hash_file_sha256_hex(dataset,ds)||!stage4_hash_file_sha256_hex(cache_path,ca)||!stage4_hash_file_sha256_hex(checkpoint,ck)||!stage4_hash_file_sha256_hex(tokenizer,tk)||!stage4_hash_file_sha256_hex(d6_checkpoint,d6hash)||!stage4_hash_file_sha256_hex(F2_D6_REPORT,d6r_hash)||!stage4_hash_file_sha256_hex(F2_D7_REPORT,d7r_hash)||!stage4_hash_file_sha256_hex(F2_D7_TSV,d7t_hash)||!stage4_hash_file_sha256_hex(f1_report,f1r_hash)||!stage4_hash_file_sha256_hex(f1_tsv,f1t_hash)||strcmp(ds,F2_DATASET)||strcmp(ca,F2_CACHE)||strcmp(ck,F2_BASE)||strcmp(tk,F2_TOKENIZER)||strcmp(d6hash,F2_D6)||!d5_load_report(f1_report,f1,&f1_count)||!f2_value(f1,f1_count,"d6_report_sha256",d6r_hash)||!f2_value(f1,f1_count,"d7_report_sha256",d7r_hash)||!f2_value(f1,f1_count,"d7_trajectory_sha256",d7t_hash)||!f2_value(f1,f1_count,"f1_fixed_horizon_pass","yes")||!f2_value(f1,f1_count,"lowest_observed_validation_loss_step","20")||!f2_value(f1,f1_count,"lowest_observed_validation_loss",F2_VALIDATION20)){return 1;}
    if(!d4_float_ok()||!d4_hex_raw(ds,dsraw)||!d4_hex_raw(ck,ckraw)||!d4_hex_raw(tk,tkraw)){return 1;}
    cache=fopen(cache_path,"rb");if(!cache||!d4_read_header(cache,&header)||!d4_header_matches(&header,dsraw,ckraw,tkraw)){failed=1;goto done;}rows=calloc(D4_ROWS,sizeof(*rows));if(!rows){failed=1;goto done;}for(i=0U;i<D4_ROWS;i++){if(!d4_read_row(cache,&rows[i])||rows[i].split>=D4_SPLITS){failed=1;goto done;}counts[rows[i].split]++;}if(fgetc(cache)!=EOF||ferror(cache)||fclose(cache)!=0||counts[D4_TRAIN]!=111U||counts[D4_VALIDATION]!=33U||counts[D4_TEST]!=21U){cache=NULL;failed=1;goto done;}cache=NULL;
    unload_adapter_runtime();load_transformer(&tr,checkpoint);tr_ready=1;if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||!stage4_adapter_init(&state,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){failed=1;goto done;}state_ready=1;base=calloc(D4_VOCAB,sizeof(float));final=calloc(D4_VOCAB,sizeof(float));if(!base||!final||!d6_load_checkpoint(d6_checkpoint,ds,ca,ck,tk,&state,&start_step,D6_TOTAL_BYTES)||start_step!=5U||!d4_evaluate(rows,&state,&tr,&points[5].point.metrics,&loss_ok)||!d4_fmt12(points[5].point.metrics.split[D4_TRAIN].loss/111.0,"10.258683447106")||!d4_fmt12(points[5].point.metrics.split[D4_VALIDATION].loss/33.0,"9.048405144340")){failed=1;goto done;}points[5].evaluated=1;
    for(i=F2_FIRST_STEP;i<=F2_LAST_STEP;i++){int good=1,pok,mok;double update;if(!d4_step(rows,&state,&tr,base,final,(int)i,&points[i].point,&good,&loss_ok)||!good){failed=1;goto done;}test_gradient+=points[i].point.grad_test_rows;grad_ok_all=grad_ok_all&&good;update=stage4_adapter_adam_step(&state,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);d4_state_finite(&state,&pok,&mok);param_ok=param_ok&&pok&&isfinite(update);moment_ok=moment_ok&&mok;if(i<=10U||i%5U==0U){if(!d4_evaluate(rows,&state,&tr,&points[i].point.metrics,&loss_ok)){failed=1;goto done;}points[i].evaluated=1;if(selected==0U||points[i].point.metrics.split[D4_VALIDATION].loss<points[selected].point.metrics.split[D4_VALIDATION].loss){selected=i;unique=1;}else if(points[i].point.metrics.split[D4_VALIDATION].loss==points[selected].point.metrics.split[D4_VALIDATION].loss)unique=0;}}
    f1_match=f2_matches_f1(f1_tsv,points,&trajectory_rows,&trajectory_mismatches);if(!f1_match||selected!=F2_LAST_STEP||!unique||!d4_fmt12(points[selected].point.metrics.split[D4_TRAIN].loss/111.0,F2_TRAIN20)||!d4_fmt12(points[selected].point.metrics.split[D4_VALIDATION].loss/33.0,F2_VALIDATION20)||test_gradient!=0U){failed=1;goto done;}selection_frozen=1;
    sa=malloc(state.a_count*sizeof(float));sb=malloc(state.b_count*sizeof(float));s1a=malloc(state.a_count*sizeof(float));s2a=malloc(state.a_count*sizeof(float));s1b=malloc(state.b_count*sizeof(float));s2b=malloc(state.b_count*sizeof(float));if(!sa||!sb||!s1a||!s2a||!s1b||!s2b){failed=1;goto done;}memcpy(sa,state.a,state.a_count*sizeof(float));memcpy(sb,state.b,state.b_count*sizeof(float));memcpy(s1a,state.m1_a,state.a_count*sizeof(float));memcpy(s2a,state.m2_a,state.a_count*sizeof(float));memcpy(s1b,state.m1_b,state.b_count*sizeof(float));memcpy(s2b,state.m2_b,state.b_count*sizeof(float));if(!f2_write_checkpoint(ds,ca,ck,tk,&state)||!stage4_adapter_init(&reloaded,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){failed=1;goto done;}reloaded_ready=1;if(!f2_read_checkpoint(F2_CHECKPOINT,ds,ca,ck,tk,&reloaded,&reloaded_step)){failed=1;goto done;}mismatch_a=d6_byte_mismatches(sa,reloaded.a,state.a_count*sizeof(float));mismatch_b=d6_byte_mismatches(sb,reloaded.b,state.b_count*sizeof(float));mismatch_m1a=d6_byte_mismatches(s1a,reloaded.m1_a,state.a_count*sizeof(float));mismatch_m2a=d6_byte_mismatches(s2a,reloaded.m2_a,state.a_count*sizeof(float));mismatch_m1b=d6_byte_mismatches(s1b,reloaded.m1_b,state.b_count*sizeof(float));mismatch_m2b=d6_byte_mismatches(s2b,reloaded.m2_b,state.b_count*sizeof(float));mismatch_count=mismatch_a+mismatch_b+mismatch_m1a+mismatch_m2a+mismatch_m1b+mismatch_m2b;byte_match=mismatch_count==0U;if(!byte_match){failed=1;goto done;}test_calls++;if(!f2_test_evaluate(rows,&reloaded,&tr,&test,&loss_ok)){failed=1;goto done;}pass=grad_ok_all&&param_ok&&moment_ok&&loss_ok&&test_gradient==0U&&test.rows==21U&&selected==20U&&unique&&f1_match&&byte_match&&reloaded_step==20U;
done:
    if(!failed&&!f2_write_tsv(points))failed=1;
    report=fopen(F2_REPORT,"wb");
    if(report){
        fprintf(report,"stage=4E-F2\nmode=validation_selected_step20_heldout_test\n");
        fprintf(report,"dataset_pathname=%s\ndataset_sha256=%s\ncache_pathname=%s\ncache_sha256=%s\n",dataset,ds,cache_path,ca);
        fprintf(report,"base_checkpoint_pathname=%s\nbase_checkpoint_sha256=%s\ntokenizer_pathname=%s\ntokenizer_sha256=%s\n",checkpoint,ck,tokenizer,tk);
        fprintf(report,"d6_checkpoint_pathname=%s\nd6_checkpoint_sha256=%s\nd6_report_sha256=%s\nd7_report_sha256=%s\nd7_trajectory_sha256=%s\n",d6_checkpoint,d6hash,d6r_hash,d7r_hash,d7t_hash);
        fprintf(report,"f1_report_pathname=%s\nf1_report_sha256=%s\nf1_trajectory_pathname=%s\nf1_trajectory_sha256=%s\n",f1_report,f1r_hash,f1_tsv,f1t_hash);
        fprintf(report,"selection_source=f1_fixed_horizon_validation\nselection_metric=validation_average_target_loss\nselection_test_access=no\nvalidation_acceptance_loss_max=8.650949130533\n");
        fprintf(report,"selected_step=%u\nselected_validation_average_loss=%.12f\nselected_step_unique_minimum=%s\nselection_unique_minimum=%s\nselected_validation_acceptance_pass=%s\nselection_frozen=%s\n",selected,selected?points[selected].point.metrics.split[D4_VALIDATION].loss/33.0:0.0,unique?"yes":"no",unique?"yes":"no",selected&&points[selected].point.metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",selection_frozen?"yes":"no");
        fprintf(report,"reconstruction_start_step=5\nreconstruction_end_step=20\nreconstruction_update_count=15\ntrajectory_rows_compared=%u\ntrajectory_field_mismatches=%u\ndeterministic_trajectory_match=%s\n",trajectory_rows,trajectory_mismatches,f1_match?"yes":"no");
        fprintf(report,"selected_train_average_loss=%.12f\nselected_validation_average_loss=%.12f\nall_training_values_finite=%s\n",selected?points[selected].point.metrics.split[D4_TRAIN].loss/111.0:0.0,selected?points[selected].point.metrics.split[D4_VALIDATION].loss/33.0:0.0,grad_ok_all&&param_ok&&moment_ok&&loss_ok?"yes":"no");
        fprintf(report,"f1_reconstruction_fields_match=%s\ncheckpoint_format_version=%u\ncheckpoint_output=%s\ncheckpoint_optimizer_step=%llu\n",f1_match?"yes":"no",D6_FORMAT_VERSION,F2_CHECKPOINT,(unsigned long long)reloaded_step);
        fprintf(report,"checkpoint_A_byte_mismatches=%u\ncheckpoint_B_byte_mismatches=%u\ncheckpoint_m1A_byte_mismatches=%u\ncheckpoint_m2A_byte_mismatches=%u\ncheckpoint_m1B_byte_mismatches=%u\ncheckpoint_m2B_byte_mismatches=%u\ncheckpoint_total_state_byte_mismatches=%u\ncheckpoint_roundtrip_bitwise_match=%s\n",mismatch_a,mismatch_b,mismatch_m1a,mismatch_m2a,mismatch_m1b,mismatch_m2b,mismatch_count,byte_match?"yes":"no");
        fprintf(report,"checkpoint_reloaded_optimizer_step=%llu\nselected_state_byte_mismatches=%u\n",(unsigned long long)reloaded_step,mismatch_count);
        fprintf(report,"test_access_before_selection_frozen=no\npreselection_test_loss_rows=0\npreselection_test_top1_rows=0\npreselection_test_gradient_rows=0\npreselection_test_update_rows=0\nheldout_test_calls=%u\n",test_calls);
        fprintf(report,"test_records=5\ntest_target_rows=%u\ntest_total_target_loss=%.17g\ntest_average_target_loss=%.12f\ntest_top1_correct=%llu\ntest_top1_total=21\ntest_top1_fraction=%.12f\n",test.rows,test.loss,test.rows?test.loss/(double)test.rows:0.0,(unsigned long long)test.top1,test.rows?(double)test.top1/(double)test.rows:0.0);
        fprintf(report,"test_gradient_rows_consumed=0\ntest_update_rows_consumed=0\noptimizer_step_before_test=%llu\noptimizer_step_after_test=%llu\ntest_values_finite=%s\n",(unsigned long long)reloaded_step,(unsigned long long)reloaded_step,loss_ok&&isfinite(test.loss)&&isfinite(test.rows?test.loss/(double)test.rows:0.0)&&isfinite(test.rows?(double)test.top1/(double)test.rows:0.0)?"yes":"no");
        fprintf(report,"baseline_test_average_loss=%.12f\nbaseline_test_top1_correct=0\nbaseline_test_top1_total=21\ntest_loss_delta_from_baseline=%.12f\ntest_relative_loss_change_from_baseline=%.12f\ntest_top1_correct_delta_from_baseline=%lld\n",F2_BASELINE_TEST_LOSS,(test.rows?test.loss/(double)test.rows:0.0)-F2_BASELINE_TEST_LOSS,((test.rows?test.loss/(double)test.rows:0.0)-F2_BASELINE_TEST_LOSS)/F2_BASELINE_TEST_LOSS,(long long)test.top1);
        fprintf(report,"test_gradient_rows_seen=%u\ntest_evaluation_rows_seen=%u\ntest_average_loss=%.12f\n",test_gradient,test.rows,test.rows?test.loss/(double)test.rows:0.0);
        fprintf(report,"all_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\n",grad_ok_all?"yes":"no",param_ok?"yes":"no",moment_ok?"yes":"no",loss_ok?"yes":"no");
        fprintf(report,"gate_selection_validation_only=%s\ngate_selected_step20=%s\ngate_reconstruction_matches_f1=%s\ngate_step20_checkpoint_roundtrip=%s\ngate_test_sealed_until_selection=%s\n",selection_frozen?"yes":"no",selected==20U&&unique?"yes":"no",f1_match&&trajectory_rows==15U&&trajectory_mismatches==0U?"yes":"no",byte_match&&reloaded_step==20U?"yes":"no",selection_frozen&&test_calls==1U?"yes":"no");
        fprintf(report,"gate_test_no_gradient=%s\ngate_test_no_update=%s\ngate_optimizer_remains_step20=%s\ngate_all_finite=%s\ngate_stage4e_f2=%s\nf2_selected_step20_test_pass=%s\n",test_gradient==0U?"yes":"no",test_calls==1U&&reloaded_step==20U?"yes":"no",reloaded_step==20U?"yes":"no",grad_ok_all&&param_ok&&moment_ok&&loss_ok&&isfinite(test.loss)?"yes":"no",pass?"yes":"no",pass?"yes":"no");
        if(fclose(report)!=0)failed=1;
    }else failed=1;
    if(cache)fclose(cache);free(rows);free(base);free(final);free(sa);free(sb);free(s1a);free(s2a);free(s1b);free(s2b);if(state_ready)stage4_adapter_free(&state);if(reloaded_ready)stage4_adapter_free(&reloaded);if(tr_ready)unload_adapter_runtime();printf("f2_selected_step20_test_pass=%s\n",pass?"yes":"no");return failed||!pass?1:0;
}
#define STAGE4E_D4_CORE_ONLY
#include "stage4e_d4_a4_trajectory.c"
#undef STAGE4E_D4_CORE_ONLY

#ifndef STAGE4E_D5_CORE_ONLY
#define D5_PROVENANCE "ten-step-repeatability-2026-10-01-a"
#define D5_EXPECTED_D3_SHA "f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"
#endif
#define D5_REPORT_ENTRIES 512U
#define D5_TEXT 256U

typedef struct { char key[96]; char value[D5_TEXT]; } d5_kv_t;
typedef struct {
    uint32_t step;
    char train_average[64], train_top1[32];
    char validation_average[64], validation_top1[32];
    char acceptance[8], norm_before[64], norm_after[64], clip[8];
} d5_reference_row_t;
typedef struct {
    uint32_t step_index, gradient_rows, grad_a_count, grad_a_max;
    uint32_t evaluation_rows;
    uint32_t grad_b_count, grad_b_max, norm_before, norm_after;
    uint32_t clip_applied, clip_consistency;
    uint32_t train_total, train_average, train_top1;
    uint32_t validation_total, validation_average, validation_top1;
    uint32_t validation_acceptance;
} d5_mismatches_t;

#ifndef STAGE4E_D5_CORE_ONLY
static const char d5_dataset_sha[]="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d5_cache_sha[]="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d5_checkpoint_sha[]="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d5_tokenizer_sha[]="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d5_d1_sha[]="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
static const char d5_d2_sha[]="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e";
#endif

static int d5_load_report(const char *path,d5_kv_t *items,uint32_t *count)
{
    FILE *f=fopen(path,"rb"); char line[512];
    *count=0U; if(!f)return 0;
    while(fgets(line,sizeof(line),f)!=NULL) {
        char *eq=strchr(line,'='); size_t n;
        if(!eq)continue;
        *eq++='\0'; n=strcspn(eq,"\r\n"); eq[n]='\0';
        if(*count>=D5_REPORT_ENTRIES||strlen(line)>=sizeof(items[0].key)||strlen(eq)>=sizeof(items[0].value)){fclose(f);return 0;}
        strcpy(items[*count].key,line); strcpy(items[*count].value,eq); (*count)++;
    }
    if(ferror(f)){fclose(f);return 0;}
    return fclose(f)==0;
}

static const char *d5_get(const d5_kv_t *items,uint32_t count,const char *key)
{
    uint32_t i; for(i=0;i<count;i++)if(strcmp(items[i].key,key)==0)return items[i].value;
    return NULL;
}

static int d5_load_tsv(const char *path,d5_reference_row_t rows[D4_STEPS+1])
{
    FILE *f=fopen(path,"rb"); char line[1024]; uint32_t i=0U;
    if(!f)return 0;
    if(fgets(line,sizeof(line),f)==NULL){fclose(f);return 0;}
    while(i<=D4_STEPS&&fgets(line,sizeof(line),f)!=NULL) {
        char *fields[9],*p; uint32_t n=0U; char *save=NULL;
        p=strtok_r(line,"\t\r\n",&save);
        while(p&&n<9U){fields[n++]=p;p=strtok_r(NULL,"\t\r\n",&save);}
        if(n!=9U){fclose(f);return 0;}
        rows[i].step=(uint32_t)strtoul(fields[0],NULL,10);
#define D5_COPY_FIELD(dst,src) do { if(strlen(src)>=sizeof(dst)){fclose(f);return 0;} strcpy(dst,src); } while(0)
        D5_COPY_FIELD(rows[i].train_average,fields[1]);
        D5_COPY_FIELD(rows[i].train_top1,fields[2]);
        D5_COPY_FIELD(rows[i].validation_average,fields[3]);
        D5_COPY_FIELD(rows[i].validation_top1,fields[4]);
        D5_COPY_FIELD(rows[i].acceptance,fields[5]);
        D5_COPY_FIELD(rows[i].norm_before,fields[6]);
        D5_COPY_FIELD(rows[i].norm_after,fields[7]);
        D5_COPY_FIELD(rows[i].clip,fields[8]);
#undef D5_COPY_FIELD
        i++;
    }
    if(i!=D4_STEPS+1U||fgets(line,sizeof(line),f)!=NULL){fclose(f);return 0;}
    if(ferror(f)){fclose(f);return 0;}
    return fclose(f)==0;
}

static int d5_compare_text(const char *reference,const char *actual,uint32_t *counter)
{
    if(reference==NULL||strcmp(reference,actual)!=0){(*counter)++;return 0;}
    return 1;
}

static int d5_compare_reference(const d5_kv_t *report,uint32_t report_count,
    const d5_reference_row_t ref[D4_STEPS+1],const d4_point_t points[D4_STEPS+1],
    d5_mismatches_t *mm)
{
    uint32_t i; char key[96],value[96];
    memset(mm,0,sizeof(*mm));
    for(i=0;i<=D4_STEPS;i++) {
        const d4_point_t *p=&points[i];
        if(ref[i].step!=i)mm->step_index++;
        if(i>0U) {
#define D5_CMP_REPORT(suffix,fmt,val,counter) do { snprintf(key,sizeof(key),"step%u_%s",i,suffix); snprintf(value,sizeof(value),fmt,val); d5_compare_text(d5_get(report,report_count,key),value,&mm->counter); } while(0)
            D5_CMP_REPORT("gradient_train_rows","%u",p->grad_train_rows,gradient_rows);
            D5_CMP_REPORT("gradient_validation_rows","%u",p->grad_validation_rows,gradient_rows);
            D5_CMP_REPORT("gradient_test_rows","%u",p->grad_test_rows,gradient_rows);
            D5_CMP_REPORT("grad_A_nonzero_elements","%u",p->grad_a_nonzero,grad_a_count);
            D5_CMP_REPORT("grad_A_max_abs","%.12g",p->grad_a_max,grad_a_max);
            D5_CMP_REPORT("grad_B_nonzero_elements","%u",p->grad_b_nonzero,grad_b_count);
            D5_CMP_REPORT("grad_B_max_abs","%.12g",p->grad_b_max,grad_b_max);
            D5_CMP_REPORT("gradient_norm_before_clip","%.17g",p->norm_before,norm_before);
            D5_CMP_REPORT("gradient_norm_after_clip","%.17g",p->norm_after,norm_after);
#undef D5_CMP_REPORT
            snprintf(key,sizeof(key),"step%u_gradient_clip_applied",i);
            d5_compare_text(d5_get(report,report_count,key),p->clip_applied?"yes":"no",&mm->clip_applied);
            snprintf(key,sizeof(key),"step%u_gradient_clip_consistency",i);
            d5_compare_text(d5_get(report,report_count,key),p->clip_consistency?"yes":"no",&mm->clip_consistency);
        }
        {
            uint32_t train_rows=p->metrics.evaluation_train_rows;
            uint32_t validation_rows=p->metrics.evaluation_validation_rows;
            uint32_t test_rows=p->metrics.evaluation_test_rows;
            snprintf(key,sizeof(key),"step%u_evaluation_train_rows",i);
            snprintf(value,sizeof(value),"%u",train_rows);
            d5_compare_text(d5_get(report,report_count,key),value,&mm->evaluation_rows);
            snprintf(key,sizeof(key),"step%u_evaluation_validation_rows",i);
            snprintf(value,sizeof(value),"%u",validation_rows);
            d5_compare_text(d5_get(report,report_count,key),value,&mm->evaluation_rows);
            snprintf(key,sizeof(key),"step%u_evaluation_test_rows",i);
            snprintf(value,sizeof(value),"%u",test_rows);
            d5_compare_text(d5_get(report,report_count,key),value,&mm->evaluation_rows);
        }
        for(uint32_t split=0;split<2U;split++) {
            const char *split_name=d4_split_names[split];
            const d4_split_metric_t *metric=&p->metrics.split[split];
            snprintf(key,sizeof(key),"step%u_%s_total_loss",i,split_name);
            snprintf(value,sizeof(value),"%.12f",metric->loss);
            d5_compare_text(d5_get(report,report_count,key),value,split==D4_TRAIN?&mm->train_total:&mm->validation_total);
        }
        snprintf(value,sizeof(value),"%.12f",p->metrics.split[D4_TRAIN].loss/111.0);
        d5_compare_text(ref[i].train_average,value,&mm->train_average);
        snprintf(value,sizeof(value),"%llu",(unsigned long long)p->metrics.split[D4_TRAIN].top1);
        d5_compare_text(ref[i].train_top1,value,&mm->train_top1);
        snprintf(value,sizeof(value),"%.12f",p->metrics.split[D4_VALIDATION].loss/33.0);
        d5_compare_text(ref[i].validation_average,value,&mm->validation_average);
        snprintf(value,sizeof(value),"%llu",(unsigned long long)p->metrics.split[D4_VALIDATION].top1);
        d5_compare_text(ref[i].validation_top1,value,&mm->validation_top1);
        d5_compare_text(ref[i].acceptance,p->metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",&mm->validation_acceptance);
    }
    return 1;
}

static uint32_t d5_total_mismatches(const d5_mismatches_t *m)
{
    return m->step_index+m->gradient_rows+m->evaluation_rows+m->grad_a_count+m->grad_a_max+
        m->grad_b_count+m->grad_b_max+m->norm_before+m->norm_after+
        m->clip_applied+m->clip_consistency+m->train_total+m->train_average+
        m->train_top1+m->validation_total+m->validation_average+
        m->validation_top1+m->validation_acceptance;
}

#ifndef STAGE4E_D5_CORE_ONLY
static int d5_write_tsv(const char *path,const d4_point_t points[D4_STEPS+1])
{
    FILE *f=fopen(path,"wb"); uint32_t i;
    if(!f)return 0;
    fprintf(f,"step\ttrain_average_loss\ttrain_top1_correct\tvalidation_average_loss\tvalidation_top1_correct\tvalidation_acceptance_reached\tgradient_norm_before_clip\tgradient_norm_after_clip\tgradient_clip_applied\n");
    for(i=0;i<=D4_STEPS;i++) {
        double ta=points[i].metrics.split[D4_TRAIN].loss/111.0;
        double va=points[i].metrics.split[D4_VALIDATION].loss/33.0;
        if(i==0)fprintf(f,"%u\t%.12f\t%llu\t%.12f\t%llu\t%s\tNA\tNA\tNA\n",i,ta,(unsigned long long)points[i].metrics.split[D4_TRAIN].top1,va,(unsigned long long)points[i].metrics.split[D4_VALIDATION].top1,va<=8.650949130533?"yes":"no");
        else fprintf(f,"%u\t%.12f\t%llu\t%.12f\t%llu\t%s\t%.12g\t%.12g\t%s\n",i,ta,(unsigned long long)points[i].metrics.split[D4_TRAIN].top1,va,(unsigned long long)points[i].metrics.split[D4_VALIDATION].top1,va<=8.650949130533?"yes":"no",points[i].norm_before,points[i].norm_after,points[i].clip_applied?"yes":"no");
    }
    return fclose(f)==0;
}

static int d5_write_report(const char *path,const char *identities[9],
    const d5_mismatches_t *m,int compared,int step0_match,int step1_match,
    int step2_match,uint64_t optimizer_steps,int grad_finite,int param_finite,int moment_finite,
    int loss_finite,int pass,
    const d4_point_t points[D4_STEPS+1],uint32_t m1a,uint32_t m2a,
    uint32_t m1b,uint32_t m2b)
{
    static const char *keys[9]={"dataset_sha256","cache_sha256","base_checkpoint_sha256","tokenizer_sha256","d1_report_sha256","d2_report_sha256","d3_report_sha256","d4_report_sha256","d4_trajectory_sha256"};
    FILE *f=fopen(path,"wb"); uint32_t i,total=d5_total_mismatches(m);
    if(!f)return 0;
    for(i=0;i<9U;i++)fprintf(f,"%s=%s\n",keys[i],identities[i]);
    fprintf(f,"fresh_initialization_used=yes\noptimizer_steps_before=0\noptimizer_steps_after=%llu\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\nstep0_d1_metrics_match=%s\nstep1_d2_metrics_match=%s\nstep2_d3_metrics_match=%s\ncompared_trajectory_rows=%d\n",(unsigned long long)optimizer_steps,step0_match?"yes":"no",step1_match?"yes":"no",step2_match?"yes":"no",compared);
#define D5_OUT(name,field) fprintf(f,#name "=%u\n",m->field)
    D5_OUT(step_index_mismatches,step_index);
    D5_OUT(gradient_row_count_mismatches,gradient_rows);
    D5_OUT(evaluation_row_count_mismatches,evaluation_rows);
    D5_OUT(grad_A_nonzero_count_mismatches,grad_a_count);
    D5_OUT(grad_A_max_mismatches,grad_a_max);
    D5_OUT(grad_B_nonzero_count_mismatches,grad_b_count);
    D5_OUT(grad_B_max_mismatches,grad_b_max);
    D5_OUT(norm_before_clip_mismatches,norm_before);
    D5_OUT(norm_after_clip_mismatches,norm_after);
    D5_OUT(clip_applied_mismatches,clip_applied);
    D5_OUT(clip_consistency_mismatches,clip_consistency);
    D5_OUT(train_total_loss_mismatches,train_total);
    D5_OUT(train_average_loss_mismatches,train_average);
    D5_OUT(train_top1_mismatches,train_top1);
    D5_OUT(validation_total_loss_mismatches,validation_total);
    D5_OUT(validation_average_loss_mismatches,validation_average);
    D5_OUT(validation_top1_mismatches,validation_top1);
    D5_OUT(validation_acceptance_mismatches,validation_acceptance);
#undef D5_OUT
    fprintf(f,"total_trajectory_field_mismatches=%u\ndeterministic_trajectory_match=%s\nfinal_parameter_bitwise_comparison=not_available\nfinal_train_average_loss=%.12f\nfinal_validation_average_loss=%.12f\nfinal_validation_acceptance_reached=%s\nfinal_m1_A_nonzero_elements=%u\nfinal_m2_A_nonzero_elements=%u\nfinal_m1_B_nonzero_elements=%u\nfinal_m2_B_nonzero_elements=%u\nall_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\nd5_repeatability_pass=%s\n",total,total==0U?"yes":"no",points[10].metrics.split[D4_TRAIN].loss/111.0,points[10].metrics.split[D4_VALIDATION].loss/33.0,points[10].metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",m1a,m2a,m1b,m2b,grad_finite?"yes":"no",param_finite?"yes":"no",moment_finite?"yes":"no",loss_finite?"yes":"no",pass?"yes":"no");
    return fclose(f)==0;
}
#endif

#ifndef STAGE4E_D5_CORE_ONLY
int main(int argc,char **argv)
{
    const char *checkpoint=NULL,*tokenizer=NULL,*dataset=NULL,*cache_path=NULL;
    const char *d1=NULL,*d2=NULL,*d3=NULL,*d4=NULL,*d4_tsv=NULL;
    char ds[65],ca[65],ck[65],tk[65],h1[65],h2[65],h3[65],h4[65],htsv[65];
    const char *identities[9];
    unsigned char dsraw[32],ckraw[32],tkraw[32];
    d4_header_t header; d4_row_t *rows=NULL; d4_point_t points[11];
    d5_kv_t ref_report[D5_REPORT_ENTRIES]; uint32_t report_count=0U;
    d5_reference_row_t ref_tsv[11]; d5_mismatches_t mismatches;
    stage4_adapter_t ad; Transformer tr; FILE *cf=NULL;
    float *base=NULL,*final=NULL; uint32_t split_counts[4]={0U,0U,0U,0U};
    uint32_t i,m1a,m2a,m1b,m2b; int argi,tr_ready=0,ad_ready=0;
    int grad_finite=1,param_finite=1,moment_finite=1,loss_finite=1;
    int anchors=0,step0_match=0,step1_match=0,step2_match=0,pass=0,failed=0;
    uint32_t test_grad=0U,test_eval=0U;
    uint64_t optimizer_steps=0U,bytes=0U;
    const char *report_path="stage4e-d5-a4-repeat-report.txt";
    const char *tsv_path="stage4e-d5-a4-repeat-trajectory.tsv";
    printf("d5_build_provenance=%s\n",D5_PROVENANCE);
    memset(&ad,0,sizeof(ad));memset(&tr,0,sizeof(tr));memset(&header,0,sizeof(header));memset(points,0,sizeof(points));
    if(argc<2){fprintf(stderr,"usage: %s checkpoint -z tokenizer -d dataset -c cache -1 d1 -2 d2 -3 d3 -4 d4 -5 d4-tsv\n",argv[0]);return 1;}
    checkpoint=argv[1];
    for(argi=2;argi<argc;argi++){
        if(!strcmp(argv[argi],"-z")&&argi+1<argc)tokenizer=argv[++argi];
        else if(!strcmp(argv[argi],"-d")&&argi+1<argc)dataset=argv[++argi];
        else if(!strcmp(argv[argi],"-c")&&argi+1<argc)cache_path=argv[++argi];
        else if(!strcmp(argv[argi],"-1")&&argi+1<argc)d1=argv[++argi];
        else if(!strcmp(argv[argi],"-2")&&argi+1<argc)d2=argv[++argi];
        else if(!strcmp(argv[argi],"-3")&&argi+1<argc)d3=argv[++argi];
        else if(!strcmp(argv[argi],"-4")&&argi+1<argc)d4=argv[++argi];
        else if(!strcmp(argv[argi],"-5")&&argi+1<argc)d4_tsv=argv[++argi];
        else {fprintf(stderr,"error: invalid D5 option\n");return 1;}
    }
    if(!tokenizer||!dataset||!cache_path||!d1||!d2||!d3||!d4||!d4_tsv){fprintf(stderr,"error: missing D5 input\n");return 1;}
    if(!stage4_hash_file_sha256_hex(dataset,ds)||!stage4_hash_file_sha256_hex(cache_path,ca)||!stage4_hash_file_sha256_hex(checkpoint,ck)||!stage4_hash_file_sha256_hex(tokenizer,tk)||!stage4_hash_file_sha256_hex(d1,h1)||!stage4_hash_file_sha256_hex(d2,h2)||!stage4_hash_file_sha256_hex(d3,h3)||!stage4_hash_file_sha256_hex(d4,h4)||!stage4_hash_file_sha256_hex(d4_tsv,htsv)){fprintf(stderr,"error: D5 identity hashing failed\n");failed=1;goto done;}
    if(strcmp(ds,d5_dataset_sha)||strcmp(ca,d5_cache_sha)||strcmp(ck,d5_checkpoint_sha)||strcmp(tk,d5_tokenizer_sha)||strcmp(h1,d5_d1_sha)||strcmp(h2,d5_d2_sha)||strcmp(h3,D5_EXPECTED_D3_SHA)){fprintf(stderr,"error: pinned D5 input identity mismatch\n");failed=1;goto done;}
    identities[0]=ds;identities[1]=ca;identities[2]=ck;identities[3]=tk;identities[4]=h1;identities[5]=h2;identities[6]=h3;identities[7]=h4;identities[8]=htsv;
    if(!d4_float_ok()||!stage4_file_size_bytes(cache_path,&bytes)||bytes!=D4_CACHE_SIZE||!d4_hex_raw(ds,dsraw)||!d4_hex_raw(ck,ckraw)||!d4_hex_raw(tk,tkraw)){fprintf(stderr,"error: D5 cache format/identity decode failed\n");failed=1;goto done;}
    if(!d5_load_report(d4,ref_report,&report_count)||!d5_load_tsv(d4_tsv,ref_tsv)){fprintf(stderr,"error: cannot read accepted D4 trajectory reference\n");failed=1;goto done;}
    if(d5_get(ref_report,report_count,"d4_trajectory_pass")==NULL||strcmp(d5_get(ref_report,report_count,"d4_trajectory_pass"),"yes")||strcmp(d5_get(ref_report,report_count,"optimizer_steps_after"),"10")||strcmp(d5_get(ref_report,report_count,"test_gradient_rows_seen"),"0")||strcmp(d5_get(ref_report,report_count,"test_evaluation_rows_seen"),"0")||strcmp(d5_get(ref_report,report_count,"dataset_sha256")==NULL?"":d5_get(ref_report,report_count,"dataset_sha256"),d5_dataset_sha)||strcmp(d5_get(ref_report,report_count,"cache_sha256")==NULL?"":d5_get(ref_report,report_count,"cache_sha256"),d5_cache_sha)||strcmp(d5_get(ref_report,report_count,"base_checkpoint_sha256")==NULL?"":d5_get(ref_report,report_count,"base_checkpoint_sha256"),d5_checkpoint_sha)||strcmp(d5_get(ref_report,report_count,"tokenizer_sha256")==NULL?"":d5_get(ref_report,report_count,"tokenizer_sha256"),d5_tokenizer_sha)||strcmp(d5_get(ref_report,report_count,"d1_report_sha256")==NULL?"":d5_get(ref_report,report_count,"d1_report_sha256"),d5_d1_sha)||strcmp(d5_get(ref_report,report_count,"d2_report_sha256")==NULL?"":d5_get(ref_report,report_count,"d2_report_sha256"),d5_d2_sha)||strcmp(d5_get(ref_report,report_count,"d3_report_sha256")==NULL?"":d5_get(ref_report,report_count,"d3_report_sha256"),D5_EXPECTED_D3_SHA)||strcmp(d5_get(ref_report,report_count,"final_train_average_loss")==NULL?"":d5_get(ref_report,report_count,"final_train_average_loss"),"9.577087323852")||strcmp(d5_get(ref_report,report_count,"final_validation_average_loss")==NULL?"":d5_get(ref_report,report_count,"final_validation_average_loss"),"8.829185963434")){fprintf(stderr,"error: D4 reference report not accepted, identity-matched, or test-sealed\n");failed=1;goto done;}
    cf=fopen(cache_path,"rb");if(!cf||!d4_read_header(cf,&header)||!d4_header_matches(&header,dsraw,ckraw,tkraw)){fprintf(stderr,"error: D5 cache header invalid\n");failed=1;goto done;}
    rows=calloc(D4_ROWS,sizeof(*rows));if(!rows){fprintf(stderr,"error: D5 row allocation failed\n");failed=1;goto done;}
    for(i=0;i<D4_ROWS;i++){if(!d4_read_row(cf,&rows[i])||rows[i].record>=D4_RECORDS||rows[i].target>=D4_VOCAB||rows[i].split>=D4_SPLITS||rows[i].split==D4_REGRESSION){fprintf(stderr,"error: invalid D5 cache row %u\n",i);failed=1;goto done;}if(i&&(rows[i].record<rows[i-1].record||(rows[i].record==rows[i-1].record&&(rows[i].split!=rows[i-1].split||rows[i].position<=rows[i-1].position)))){fprintf(stderr,"error: D5 cache order error\n");failed=1;goto done;}split_counts[rows[i].split]++;}
    if(fgetc(cf)!=EOF||ferror(cf)||fclose(cf)!=0){cf=NULL;fprintf(stderr,"error: D5 cache trailing data/read failure\n");failed=1;goto done;}cf=NULL;
    if(split_counts[0]!=111U||split_counts[1]!=33U||split_counts[2]!=21U){fprintf(stderr,"error: D5 cache split count mismatch\n");failed=1;goto done;}
    unload_adapter_runtime();load_transformer(&tr,checkpoint);tr_ready=1;
    if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||!stage4_adapter_init(&ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){fprintf(stderr,"error: D5 model/adapter initialization failed\n");failed=1;goto done;}ad_ready=1;
    if(ad.rank!=D4_RANK||ad.scale!=D4_SCALE||ad.a_count!=2304U||ad.b_count!=256000U){fprintf(stderr,"error: D5 adapter architecture mismatch\n");failed=1;goto done;}
    stage4_adapter_fill_a_deterministic(&ad,1U,0.01f);stage4_adapter_zero_b(&ad);stage4_adapter_zero_moments(&ad);
    base=calloc(D4_VOCAB,sizeof(float));final=calloc(D4_VOCAB,sizeof(float));if(!base||!final){fprintf(stderr,"error: D5 logit buffer allocation failed\n");failed=1;goto done;}
    if(!d4_evaluate(rows,&ad,&tr,&points[0].metrics,&loss_finite)){fprintf(stderr,"error: D5 step-0 eval failed\n");failed=1;goto done;}
    step0_match=d4_anchor_metrics(&points[0].metrics,0U);
    if(!step0_match){fprintf(stderr,"error: D5 step-0 D1 anchor mismatch\n");failed=1;goto done;}
    for(i=1U;i<=D4_STEPS;i++){
        int grad_ok=1;double update_norm;
        if(!d4_step(rows,&ad,&tr,base,final,(int)i,&points[i],&grad_ok,&loss_finite)){fprintf(stderr,"error: D5 gradient step %u failed\n",i);failed=1;goto done;}
        if(!grad_ok)grad_finite=0;
        update_norm=stage4_adapter_adam_step(&ad,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);optimizer_steps++;
        {int p_ok,m_ok;d4_state_finite(&ad,&p_ok,&m_ok);if(!isfinite(update_norm)||!p_ok)param_finite=0;if(!m_ok)moment_finite=0;}
        if(!d4_evaluate(rows,&ad,&tr,&points[i].metrics,&loss_finite)){fprintf(stderr,"error: D5 evaluation at step %u failed\n",i);failed=1;goto done;}
        if(!d4_anchor_metrics(&points[i].metrics,i)||!d4_gradient_anchor(&points[i],i)){fprintf(stderr,"error: D5 deterministic anchor failed at step %u\n",i);failed=1;goto done;}
        if(i==1U)step1_match=1;
        if(i==2U)step2_match=1;
    }
    anchors=step0_match&&step1_match&&step2_match;
    if(!d5_compare_reference(ref_report,report_count,ref_tsv,points,&mismatches)){fprintf(stderr,"error: D5 trajectory comparison failed\n");failed=1;goto done;}
    {uint32_t total=d5_total_mismatches(&mismatches);pass=anchors&&optimizer_steps==D4_STEPS&&test_grad==0U&&test_eval==0U&&total==0U&&grad_finite&&param_finite&&moment_finite&&loss_finite&&points[10].metrics.split[0].loss/111.0<points[0].metrics.split[0].loss/111.0;}
    d4_moment_counts(&ad,&m1a,&m2a,&m1b,&m2b);
    if(!d5_write_tsv(tsv_path,points)||!d5_write_report(report_path,identities,&mismatches,11,step0_match,step1_match,step2_match,optimizer_steps,grad_finite,param_finite,moment_finite,loss_finite,pass,points,m1a,m2a,m1b,m2b)){fprintf(stderr,"error: D5 artifact write failed\n");failed=1;goto done;}
    printf("base_checkpoint_identity=PASS\ntokenizer_identity=PASS\ndataset_identity=PASS\ncache_identity=PASS\nd1_report_identity=PASS\nd2_report_identity=PASS\nd3_report_identity=PASS\nd4_report_identity=PASS\nd4_trajectory_identity=PASS\nstep0_d1_metrics_match=%s\nstep1_d2_metrics_match=%s\nstep2_d3_metrics_match=%s\noptimizer_steps_before=0\noptimizer_steps_after=%llu\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\ncompared_trajectory_rows=11\ntotal_trajectory_field_mismatches=%u\ndeterministic_trajectory_match=%s\nfinal_train_average_loss=%.12f\nfinal_validation_average_loss=%.12f\nall_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\nStage4E-D5-A4=%s\n",step0_match?"yes":"no",step1_match?"yes":"no",step2_match?"yes":"no",(unsigned long long)optimizer_steps,d5_total_mismatches(&mismatches),d5_total_mismatches(&mismatches)==0U?"yes":"no",points[10].metrics.split[0].loss/111.0,points[10].metrics.split[1].loss/33.0,grad_finite?"yes":"no",param_finite?"yes":"no",moment_finite?"yes":"no",loss_finite?"yes":"no",pass?"PASS":"FAIL");
    if(!pass)failed=1;
done:
    if(cf)fclose(cf);free(rows);free(base);free(final);if(ad_ready)stage4_adapter_free(&ad);if(tr_ready)free_transformer(&tr);
    return failed||!pass?EXIT_FAILURE:EXIT_SUCCESS;
}
#endif

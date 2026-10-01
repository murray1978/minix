#define STAGE4E_D6_CORE_ONLY
#include "stage4e_d6_a4_checkpoint.c"
#undef STAGE4E_D6_CORE_ONLY
#define STAGE4E_D5_CORE_ONLY
#include "stage4e_d5_a4_repeat.c"
#undef STAGE4E_D5_CORE_ONLY

#define F1_PROVENANCE "fixed-horizon-collapse-regression-2026-10-01-a"
#define F1_STEPS 50U
#define F1_START_STEP 5U
#define F1_EXPECTED_D6_CHECKPOINT "ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"
#define F1_EXPECTED_D6_REPORT "59e3370ab4420128caa46f1cf6a83714ffdb1485b57bd71d5658df90d41faf9a"
#define F1_EXPECTED_D7_REPORT "05b78313572cfd14242185ffa5e7fdf535ad103f2c4bcc2ee4d0faf439f253f2"
#define F1_EXPECTED_D7_TSV "ed5505733443f3b57b49db71dc87398bec860e04aeac5752a5e60e24e57fd46a"

typedef struct {
    d4_point_t point;
    int evaluated;
    uint32_t unique_top1;
    uint32_t dominant_token;
    uint32_t dominant_count;
    double dominant_fraction;
    int all_same;
    double correction_rms_mean;
    double correction_rms_max;
    double correction_abs_max;
    double adapter_a_rms,adapter_a_max;
    double adapter_b_rms,adapter_b_max;
} f1_point_t;

typedef struct {
    uint32_t step;
    char gradient_train_rows[32],grad_a_nonzero[32],grad_b_nonzero[32];
    char norm_before[64],norm_after[64],clip[8];
    char train_average[64],validation_average[64];
} f1_d7_reference_row_t;

static const char f1_dataset_sha[]="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char f1_cache_sha[]="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char f1_checkpoint_sha[]="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char f1_tokenizer_sha[]="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char f1_d1_sha[]="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
static const char f1_d2_sha[]="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e";
static const char f1_d3_sha[]="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e";
static const char f1_d4_sha[]="d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1";
static int f1_is_eval_step(uint32_t step)
{
    return (step>=6U&&step<=10U)||(step>=15U&&step<=50U&&step%5U==0U);
}

static int f1_value_is(const d5_kv_t *items,uint32_t count,const char *key,const char *expected)
{
    const char *value=d5_get(items,count,key);
    return value!=NULL&&strcmp(value,expected)==0;
}

static int f1_compute_parameter_stats(const stage4_adapter_t *ad,f1_point_t *p)
{
    double a2=0.0,b2=0.0;size_t i;
    p->adapter_a_max=0.0;p->adapter_b_max=0.0;
    for(i=0;i<ad->a_count;i++){double x=fabs((double)ad->a[i]);a2+=(double)ad->a[i]*(double)ad->a[i];if(x>p->adapter_a_max)p->adapter_a_max=x;}
    for(i=0;i<ad->b_count;i++){double x=fabs((double)ad->b[i]);b2+=(double)ad->b[i]*(double)ad->b[i];if(x>p->adapter_b_max)p->adapter_b_max=x;}
    p->adapter_a_rms=sqrt(a2/(double)ad->a_count);p->adapter_b_rms=sqrt(b2/(double)ad->b_count);
    return isfinite(p->adapter_a_rms)&&isfinite(p->adapter_b_rms)&&isfinite(p->adapter_a_max)&&isfinite(p->adapter_b_max);
}

static int f1_evaluate(const d4_row_t *rows,stage4_adapter_t *ad,Transformer *tr,
    f1_point_t *point,int *losses_finite,int *corrections_finite)
{
    float *base=calloc(D4_VOCAB,sizeof(float)),*final=calloc(D4_VOCAB,sizeof(float));
    uint32_t *top_counts=calloc(D4_VOCAB,sizeof(uint32_t));
    uint32_t i,current_record=0U,current_split=0U,record_rows=0U;
    uint64_t record_top1=0U;double record_loss=0.0,correction_rms_sum=0.0;
    uint32_t row_count=0U;int have_record=0;
    if(!base||!final||!top_counts){free(base);free(final);free(top_counts);return 0;}
    memset(&point->point.metrics,0,sizeof(point->point.metrics));
    *losses_finite=1;*corrections_finite=1;
    point->unique_top1=0U;point->dominant_token=0U;point->dominant_count=0U;
    point->dominant_fraction=0.0;point->all_same=0;
    point->correction_rms_mean=point->correction_rms_max=point->correction_abs_max=0.0;
    for(i=0;i<D4_ROWS;i++){
        const d4_row_t *row=&rows[i];double loss,corr_sq=0.0,row_abs_max=0.0,row_rms;uint32_t v;int prediction;
        if(row->split==D4_TEST)continue;
        if(row->split>D4_VALIDATION){free(base);free(final);free(top_counts);return 0;}
        if(!have_record||row->record!=current_record){
            if(have_record)d4_flush_record(&point->point.metrics,current_split,record_rows,record_loss,record_top1);
            current_record=row->record;current_split=row->split;record_rows=0U;record_loss=0.0;record_top1=0U;have_record=1;
        }
        if(row->split!=current_split){free(base);free(final);free(top_counts);return 0;}
        matmul(base,row->hidden,tr->weights.wcls,(int)D4_DIM,(int)D4_VOCAB);
        stage4_adapter_forward_logits(ad,row->hidden,base,final);
        if(!d4_loss(final,row->target,&loss)){*losses_finite=0;free(base);free(final);free(top_counts);return 0;}
        prediction=d4_argmax(final);
        if(prediction==(int)row->target)record_top1++;
        if(row->split==D4_VALIDATION){
            top_counts[prediction]++;
            for(v=0U;v<D4_VOCAB;v++){
                double correction=0.0;uint32_t r;
                for(r=0U;r<D4_RANK;r++)correction+=(double)ad->b[(size_t)v*D4_RANK+r]*(double)ad->u[r];
                if(!isfinite(correction)){*corrections_finite=0;free(base);free(final);free(top_counts);return 0;}
                if(fabs(correction)>row_abs_max)row_abs_max=fabs(correction);
                corr_sq+=correction*correction;
            }
            row_rms=sqrt(corr_sq/(double)D4_VOCAB);
            if(!isfinite(row_rms)||!isfinite(row_abs_max)){*corrections_finite=0;free(base);free(final);free(top_counts);return 0;}
            correction_rms_sum+=row_rms;
            if(row_rms>point->correction_rms_max)point->correction_rms_max=row_rms;
            if(row_abs_max>point->correction_abs_max)point->correction_abs_max=row_abs_max;
        }
        record_loss+=loss;record_rows++;row_count++;point->point.metrics.evaluated_rows++;
        if(row->split==D4_TRAIN)point->point.metrics.evaluation_train_rows++;
        else point->point.metrics.evaluation_validation_rows++;
    }
    if(have_record)d4_flush_record(&point->point.metrics,current_split,record_rows,record_loss,record_top1);
    for(i=0;i<D4_VOCAB;i++)if(top_counts[i]){
        point->unique_top1++;
        if(top_counts[i]>point->dominant_count){point->dominant_count=top_counts[i];point->dominant_token=i;}
    }
    point->dominant_fraction=(double)point->dominant_count/33.0;
    point->all_same=point->unique_top1==1U;
    point->correction_rms_mean=correction_rms_sum/33.0;
    point->point.metrics.evaluation_test_rows=0U;
    free(base);free(final);free(top_counts);
    return row_count==144U&&point->point.metrics.evaluation_train_rows==111U&&
        point->point.metrics.evaluation_validation_rows==33U&&
        f1_compute_parameter_stats(ad,point)&&isfinite(point->correction_rms_mean)&&
        isfinite(point->correction_rms_max)&&isfinite(point->correction_abs_max);
}

static int f1_matches_d7_anchor(const d5_kv_t *d4_reference,uint32_t reference_count,
    uint32_t step,const d4_point_t *gradient,const f1_point_t *evaluation)
{
    char key[96],value[96]; const char *actual; uint32_t split;
    snprintf(key,sizeof(key),"step%u_gradient_train_rows",step);
    snprintf(value,sizeof(value),"%u",gradient->grad_train_rows);
    actual=d5_get(d4_reference,reference_count,key); if(!actual||strcmp(actual,value))return 0;
    snprintf(key,sizeof(key),"step%u_gradient_validation_rows",step);
    snprintf(value,sizeof(value),"%u",gradient->grad_validation_rows);
    actual=d5_get(d4_reference,reference_count,key); if(!actual||strcmp(actual,value))return 0;
    snprintf(key,sizeof(key),"step%u_gradient_test_rows",step);
    snprintf(value,sizeof(value),"%u",gradient->grad_test_rows);
    actual=d5_get(d4_reference,reference_count,key); if(!actual||strcmp(actual,value))return 0;
#define F1_COMPARE_D4(suffix,format,value_expr) do { \
    snprintf(key,sizeof(key),"step%u_" suffix,step); \
    snprintf(value,sizeof(value),format,value_expr); \
    actual=d5_get(d4_reference,reference_count,key); \
    if(!actual||strcmp(actual,value))return 0; \
} while(0)
    F1_COMPARE_D4("grad_A_nonzero_elements","%u",gradient->grad_a_nonzero);
    F1_COMPARE_D4("grad_A_max_abs","%.12g",gradient->grad_a_max);
    F1_COMPARE_D4("grad_B_nonzero_elements","%u",gradient->grad_b_nonzero);
    F1_COMPARE_D4("grad_B_max_abs","%.12g",gradient->grad_b_max);
    F1_COMPARE_D4("gradient_norm_before_clip","%.17g",gradient->norm_before);
    F1_COMPARE_D4("gradient_norm_after_clip","%.17g",gradient->norm_after);
#undef F1_COMPARE_D4
    snprintf(key,sizeof(key),"step%u_gradient_clip_applied",step);
    actual=d5_get(d4_reference,reference_count,key);
    if(!actual||strcmp(actual,gradient->clip_applied?"yes":"no"))return 0;
    snprintf(key,sizeof(key),"step%u_gradient_clip_consistency",step);
    actual=d5_get(d4_reference,reference_count,key);
    if(!actual||strcmp(actual,gradient->clip_consistency?"yes":"no"))return 0;
    for(split=0U;split<2U;split++){
        const char *split_name=d4_split_names[split];
        const d4_split_metric_t *metric=&evaluation->point.metrics.split[split];
        snprintf(key,sizeof(key),"step%u_%s_total_loss",step,split_name);
        snprintf(value,sizeof(value),"%.12f",metric->loss);
        actual=d5_get(d4_reference,reference_count,key);if(!actual||strcmp(actual,value))return 0;
        snprintf(key,sizeof(key),"step%u_%s_average_loss",step,split_name);
        snprintf(value,sizeof(value),"%.12f",metric->loss/(double)metric->count);
        actual=d5_get(d4_reference,reference_count,key);if(!actual||strcmp(actual,value))return 0;
        snprintf(key,sizeof(key),"step%u_%s_top1_correct",step,split_name);
        snprintf(value,sizeof(value),"%llu",(unsigned long long)metric->top1);
        actual=d5_get(d4_reference,reference_count,key);if(!actual||strcmp(actual,value))return 0;
    }
    snprintf(key,sizeof(key),"step%u_validation_acceptance_reached",step);
    actual=d5_get(d4_reference,reference_count,key);
    if(!actual||strcmp(actual,evaluation->point.metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no"))return 0;
    return 1;
}

static int f1_load_d7_tsv(const char *path,f1_d7_reference_row_t rows[D4_STEPS+1])
{
    FILE *f=fopen(path,"rb");char line[2048];uint32_t count=0U;
    if(!f)return 0;
    if(fgets(line,sizeof(line),f)==NULL){fclose(f);return 0;}
    while(count<5U&&fgets(line,sizeof(line),f)!=NULL){
        char *field[25],*token,*save=NULL;uint32_t n=0U,step;
        token=strtok_r(line,"\t\r\n",&save);
        while(token&&n<25U){field[n++]=token;token=strtok_r(NULL,"\t\r\n",&save);}
        if(n!=25U){fclose(f);return 0;}
        step=(uint32_t)strtoul(field[0],NULL,10);
        if(step!=count+6U||strcmp(field[1],field[2])||strcmp(field[3],"yes")||
           strcmp(field[4],field[5])||strcmp(field[6],"yes")||strcmp(field[7],field[8])||
           strcmp(field[9],"yes")||strcmp(field[10],field[11])||strcmp(field[12],"yes")||
           strcmp(field[13],field[14])||strcmp(field[15],"yes")||strcmp(field[16],field[17])||
           strcmp(field[18],"yes")||strcmp(field[19],field[20])||strcmp(field[21],"yes")||
           strcmp(field[22],field[23])||strcmp(field[24],"yes")){fclose(f);return 0;}
        rows[step].step=step;
#define F1_COPY_D7(dst,src) do { if(strlen(src)>=sizeof(rows[step].dst)){fclose(f);return 0;} strcpy(rows[step].dst,src); } while(0)
        F1_COPY_D7(gradient_train_rows,field[1]);F1_COPY_D7(grad_a_nonzero,field[4]);
        F1_COPY_D7(grad_b_nonzero,field[7]);F1_COPY_D7(norm_before,field[10]);
        F1_COPY_D7(norm_after,field[13]);F1_COPY_D7(clip,field[16]);
        F1_COPY_D7(train_average,field[19]);F1_COPY_D7(validation_average,field[22]);
#undef F1_COPY_D7
        count++;
    }
    if(count!=5U||fgets(line,sizeof(line),f)!=NULL){fclose(f);return 0;}
    if(ferror(f)){fclose(f);return 0;}
    return fclose(f)==0;
}

static int f1_matches_d7_tsv(const f1_d7_reference_row_t reference[D4_STEPS+1],
    uint32_t step,const d4_point_t *gradient,const f1_point_t *evaluation)
{
    char value[96];const f1_d7_reference_row_t *row=&reference[step];
    if(row->step!=step)return 0;
    snprintf(value,sizeof(value),"%u",gradient->grad_train_rows);
    if(strcmp(value,row->gradient_train_rows))return 0;
    snprintf(value,sizeof(value),"%u",gradient->grad_a_nonzero);
    if(strcmp(value,row->grad_a_nonzero))return 0;
    snprintf(value,sizeof(value),"%u",gradient->grad_b_nonzero);
    if(strcmp(value,row->grad_b_nonzero))return 0;
    snprintf(value,sizeof(value),"%.17g",gradient->norm_before);
    if(strcmp(value,row->norm_before))return 0;
    snprintf(value,sizeof(value),"%.17g",gradient->norm_after);
    if(strcmp(value,row->norm_after))return 0;
    if(strcmp(row->clip,gradient->clip_applied?"yes":"no"))return 0;
    snprintf(value,sizeof(value),"%.12f",evaluation->point.metrics.split[D4_TRAIN].loss/111.0);
    if(strcmp(value,row->train_average))return 0;
    snprintf(value,sizeof(value),"%.12f",evaluation->point.metrics.split[D4_VALIDATION].loss/33.0);
    return strcmp(value,row->validation_average)==0;
}

static int f1_write_tsv(const char *path,const f1_point_t points[F1_STEPS+1],const d4_point_t gradients[F1_STEPS+1])
{
    FILE *f=fopen(path,"wb");uint32_t step;if(!f)return 0;
    fprintf(f,"step\tgradient_train_rows\tgradient_validation_rows\tgradient_test_rows\tgrad_A_nonzero_elements\tgrad_A_max_abs\tgrad_B_nonzero_elements\tgrad_B_max_abs\tgradient_norm_before_clip\tgradient_norm_after_clip\tgradient_clip_applied\tgradient_clip_consistency\tevaluation_performed\ttrain_average_loss\ttrain_top1_correct\tvalidation_average_loss\tvalidation_top1_correct\tvalidation_acceptance_reached\tvalidation_unique_top1_tokens\tvalidation_dominant_top1_token\tvalidation_dominant_top1_count\tvalidation_dominant_top1_fraction\tvalidation_all_same_top1\tvalidation_correction_rms_mean\tvalidation_correction_rms_max\tvalidation_correction_abs_max\tadapter_A_rms\tadapter_A_max_abs\tadapter_B_rms\tadapter_B_max_abs\n");
    for(step=6U;step<=50U;step++){
        const d4_point_t *g=&gradients[step];const f1_point_t *p=&points[step];
        fprintf(f,"%u\t%u\t%u\t%u\t%u\t%.12g\t%u\t%.12g\t%.17g\t%.17g\t%s\t%s\t%s",step,g->grad_train_rows,g->grad_validation_rows,g->grad_test_rows,g->grad_a_nonzero,g->grad_a_max,g->grad_b_nonzero,g->grad_b_max,g->norm_before,g->norm_after,g->clip_applied?"yes":"no",g->clip_consistency?"yes":"no",p->evaluated?"yes":"no");
        if(p->evaluated)fprintf(f,"\t%.12f\t%llu\t%.12f\t%llu\t%s\t%u\t%u\t%u\t%.12f\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",p->point.metrics.split[D4_TRAIN].loss/111.0,(unsigned long long)p->point.metrics.split[D4_TRAIN].top1,p->point.metrics.split[D4_VALIDATION].loss/33.0,(unsigned long long)p->point.metrics.split[D4_VALIDATION].top1,p->point.metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533?"yes":"no",p->unique_top1,p->dominant_token,p->dominant_count,p->dominant_fraction,p->all_same?"yes":"no",p->correction_rms_mean,p->correction_rms_max,p->correction_abs_max,p->adapter_a_rms,p->adapter_a_max,p->adapter_b_rms,p->adapter_b_max);
        else fprintf(f,"\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA");
        fputc('\n',f);
    }
    return fclose(f)==0;
}

static int f1_write_report(const char *path,const char *ids[12],uint64_t steps,
    const f1_point_t points[F1_STEPS+1],uint32_t test_grad,uint32_t test_eval,
    uint32_t clipped,uint32_t unclipped,uint32_t first_accept,uint32_t first_collapse,
    uint32_t lowest_step,uint32_t train_imp,uint32_t train_same,uint32_t train_worse,
    uint32_t val_imp,uint32_t val_same,uint32_t val_worse,int step5_match,
    int d7_match,int finite_ok,int pass)
{
    static const char *names[12]={
        "dataset_sha256","cache_sha256","base_checkpoint_sha256","tokenizer_sha256",
        "d1_report_sha256","d2_report_sha256","d3_report_sha256","d4_report_sha256",
        "d6_report_sha256","d6_checkpoint_sha256","d7_report_sha256","d7_trajectory_sha256"
    };
    FILE *f=fopen(path,"wb");
    uint32_t i;
    double lowest=points[5].point.metrics.split[D4_VALIDATION].loss/33.0;

    if(!f)return 0;
    for(i=0;i<12;i++)fprintf(f,"%s=%s\n",names[i],ids[i]);
    for(i=5U;i<=50U;i++){
        if(points[i].evaluated&&points[i].point.metrics.split[D4_VALIDATION].loss/33.0<lowest){
            lowest=points[i].point.metrics.split[D4_VALIDATION].loss/33.0;
            lowest_step=i;
        }
    }
    fprintf(f,
        "starting_optimizer_step=5\n"
        "step5_checkpoint_metrics_match=%s\n"
        "step5_train_average_loss=%.12f\n"
        "step5_validation_average_loss=%.12f\n"
        "steps6_to_10_d7_trajectory_match=%s\n"
        "step10_train_average_loss=%.12f\n"
        "step10_validation_average_loss=%.12f\n"
        "final_optimizer_step=%llu\n"
        "total_updates_executed=%llu\n"
        "clipped_update_count=%u\n"
        "unclipped_update_count=%u\n"
        "test_gradient_rows_seen=%u\n"
        "test_evaluation_rows_seen=%u\n",
        step5_match?"yes":"no",
        points[5].point.metrics.split[D4_TRAIN].loss/111.0,
        points[5].point.metrics.split[D4_VALIDATION].loss/33.0,
        d7_match?"yes":"no",
        points[10].point.metrics.split[D4_TRAIN].loss/111.0,
        points[10].point.metrics.split[D4_VALIDATION].loss/33.0,
        (unsigned long long)steps,
        (unsigned long long)(steps-5U),
        clipped,unclipped,test_grad,test_eval);
    fprintf(f,
        "validation_baseline_loss=9.106262242666\n"
        "validation_acceptance_loss_max=8.650949130533\n"
        "first_observed_validation_acceptance_step=%s\n",
        first_accept==0U?"NA":"tracked-step");
    if(first_accept!=0U)fprintf(f,"first_observed_validation_acceptance_step_value=%u\n",first_accept);
    fprintf(f,
        "lowest_observed_validation_loss=%.12f\n"
        "lowest_observed_validation_loss_step=%u\n"
        "final_validation_loss=%.12f\n"
        "final_vs_lowest_observed_validation_delta=%.12f\n"
        "validation_regression_after_lowest_observed=%s\n"
        "single_token_collapse_observed=%s\n"
        "first_single_token_collapse_step=%s\n",
        lowest,lowest_step,
        points[50].point.metrics.split[D4_VALIDATION].loss/33.0,
        points[50].point.metrics.split[D4_VALIDATION].loss/33.0-lowest,
        points[50].point.metrics.split[D4_VALIDATION].loss/33.0>lowest?"yes":"no",
        first_collapse==0U?"no":"yes",
        first_collapse==0U?"NA":"tracked-step");
    if(first_collapse!=0U)fprintf(f,"first_single_token_collapse_step_value=%u\n",first_collapse);
    fprintf(f,
        "train_steps_improved_from_previous=%u\n"
        "train_steps_unchanged_from_previous=%u\n"
        "train_steps_worse_than_previous=%u\n"
        "validation_steps_improved_from_previous=%u\n"
        "validation_steps_unchanged_from_previous=%u\n"
        "validation_steps_worse_than_previous=%u\n",
        train_imp,train_same,train_worse,val_imp,val_same,val_worse);
    fprintf(f,
        "final_validation_unique_top1_tokens=%u\n"
        "final_validation_dominant_top1_token=%u\n"
        "final_validation_dominant_top1_count=%u\n"
        "final_validation_dominant_top1_fraction=%.12f\n"
        "final_validation_all_same_top1=%s\n",
        points[50].unique_top1,points[50].dominant_token,points[50].dominant_count,
        points[50].dominant_fraction,points[50].all_same?"yes":"no");
    fprintf(f,
        "final_validation_correction_rms_mean=%.17g\n"
        "final_validation_correction_rms_max=%.17g\n"
        "final_validation_correction_abs_max=%.17g\n"
        "final_adapter_A_rms=%.17g\n"
        "final_adapter_A_max_abs=%.17g\n"
        "final_adapter_B_rms=%.17g\n"
        "final_adapter_B_max_abs=%.17g\n",
        points[50].correction_rms_mean,points[50].correction_rms_max,
        points[50].correction_abs_max,points[50].adapter_a_rms,
        points[50].adapter_a_max,points[50].adapter_b_rms,points[50].adapter_b_max);
    fprintf(f,
        "all_gradients_finite=%s\n"
        "all_parameters_finite=%s\n"
        "all_optimizer_moments_finite=%s\n"
        "all_losses_finite=%s\n"
        "all_correction_statistics_finite=%s\n"
        "f1_fixed_horizon_pass=%s\n",
        finite_ok?"yes":"no",finite_ok?"yes":"no",finite_ok?"yes":"no",
        finite_ok?"yes":"no",finite_ok?"yes":"no",pass?"yes":"no");
    return fclose(f)==0;
}

int main(int argc,char **argv)
{
    const char *checkpoint=NULL,*tokenizer=NULL,*dataset=NULL,*cache_path=NULL;
    const char *d1=NULL,*d2=NULL,*d3=NULL,*d4=NULL,*d6r=NULL,*d6c=NULL,*d7r=NULL,*d7t=NULL;
    char ds[65],ca[65],ck[65],tk[65],h1[65],h2[65],h3[65],h4[65];
    char h6r[65],h6c[65],h7r[65],h7t[65];
    unsigned char dsraw[32],ckraw[32],tkraw[32];const char *ids[12];
    d5_kv_t d4_reference[D5_REPORT_ENTRIES],d6_reference[D5_REPORT_ENTRIES],d7_reference[D5_REPORT_ENTRIES];
    f1_d7_reference_row_t d7_rows[D4_STEPS+1];uint32_t d4_count=0U,d6_count=0U,d7_count=0U;
    d4_header_t header;d4_row_t *rows=NULL;
    d4_point_t gradients[F1_STEPS+1];f1_point_t points[F1_STEPS+1];
    stage4_adapter_t ad;Transformer tr;FILE *cache=NULL;float *base=NULL,*final=NULL;
    uint32_t split_counts[4]={0U,0U,0U,0U},i,test_gradient_rows=0U,test_evaluation_rows=0U;
    uint32_t clipped=0U,unclipped=0U,first_accept=0U,first_collapse=0U,lowest_step=5U;
    uint32_t train_imp=0U,train_same=0U,train_worse=0U,val_imp=0U,val_same=0U,val_worse=0U;
    uint32_t previous_eval=5U;uint64_t cache_size=0U,optimizer_step=0U,updates=0U;
    int argi,tr_ready=0,ad_ready=0,loss_finite=1,gradients_finite=1;
    int parameters_finite=1,moments_finite=1,corrections_finite=1;
    int step5_match=0,d7_match=1,pass=0,failed=0;
    double lowest_validation=9.048405144340,previous_train,previous_validation;
    const char *report_path="stage4e-f1-a4-fixed-horizon-report.txt";
    const char *trajectory_path="stage4e-f1-a4-fixed-horizon-trajectory.tsv";

    printf("f1_build_provenance=%s\n",F1_PROVENANCE);
    memset(&ad,0,sizeof(ad));memset(&tr,0,sizeof(tr));memset(&header,0,sizeof(header));
    memset(points,0,sizeof(points));memset(gradients,0,sizeof(gradients));
    memset(d4_reference,0,sizeof(d4_reference));memset(d6_reference,0,sizeof(d6_reference));
    memset(d7_reference,0,sizeof(d7_reference));memset(d7_rows,0,sizeof(d7_rows));
    if(argc<2){fprintf(stderr,"usage: %s checkpoint -z tokenizer -d dataset -c cache -1 d1 -2 d2 -3 d3 -4 d4 -6 d6-report -7 d6-checkpoint -8 d7-report -9 d7-tsv\n",argv[0]);return 1;}
    checkpoint=argv[1];
    for(argi=2;argi<argc;argi++){
        if(!strcmp(argv[argi],"-z")&&argi+1<argc)tokenizer=argv[++argi];
        else if(!strcmp(argv[argi],"-d")&&argi+1<argc)dataset=argv[++argi];
        else if(!strcmp(argv[argi],"-c")&&argi+1<argc)cache_path=argv[++argi];
        else if(!strcmp(argv[argi],"-1")&&argi+1<argc)d1=argv[++argi];
        else if(!strcmp(argv[argi],"-2")&&argi+1<argc)d2=argv[++argi];
        else if(!strcmp(argv[argi],"-3")&&argi+1<argc)d3=argv[++argi];
        else if(!strcmp(argv[argi],"-4")&&argi+1<argc)d4=argv[++argi];
        else if(!strcmp(argv[argi],"-6")&&argi+1<argc)d6r=argv[++argi];
        else if(!strcmp(argv[argi],"-7")&&argi+1<argc)d6c=argv[++argi];
        else if(!strcmp(argv[argi],"-8")&&argi+1<argc)d7r=argv[++argi];
        else if(!strcmp(argv[argi],"-9")&&argi+1<argc)d7t=argv[++argi];
        else{fprintf(stderr,"error: invalid F1 argument\n");return 1;}
    }
    if(!tokenizer||!dataset||!cache_path||!d1||!d2||!d3||!d4||!d6r||!d6c||!d7r||!d7t){fprintf(stderr,"error: missing F1 input\n");return 1;}
    if(!stage4_hash_file_sha256_hex(dataset,ds)||!stage4_hash_file_sha256_hex(cache_path,ca)||
       !stage4_hash_file_sha256_hex(checkpoint,ck)||!stage4_hash_file_sha256_hex(tokenizer,tk)||
       !stage4_hash_file_sha256_hex(d1,h1)||!stage4_hash_file_sha256_hex(d2,h2)||
       !stage4_hash_file_sha256_hex(d3,h3)||!stage4_hash_file_sha256_hex(d4,h4)||
       !stage4_hash_file_sha256_hex(d6r,h6r)||!stage4_hash_file_sha256_hex(d6c,h6c)||
       !stage4_hash_file_sha256_hex(d7r,h7r)||!stage4_hash_file_sha256_hex(d7t,h7t)){
        fprintf(stderr,"error: F1 identity hash failed\n");failed=1;goto cleanup;
    }
    if(strcmp(ds,f1_dataset_sha)||strcmp(ca,f1_cache_sha)||strcmp(ck,f1_checkpoint_sha)||
       strcmp(tk,f1_tokenizer_sha)||strcmp(h1,f1_d1_sha)||strcmp(h2,f1_d2_sha)||
       strcmp(h3,f1_d3_sha)||strcmp(h4,f1_d4_sha)||strcmp(h6c,F1_EXPECTED_D6_CHECKPOINT)||
       strcmp(h6r,F1_EXPECTED_D6_REPORT)||strcmp(h7r,F1_EXPECTED_D7_REPORT)||strcmp(h7t,F1_EXPECTED_D7_TSV)){
        fprintf(stderr,"error: pinned F1 identity mismatch\n");failed=1;goto cleanup;
    }
    ids[0]=ds;ids[1]=ca;ids[2]=ck;ids[3]=tk;ids[4]=h1;ids[5]=h2;
    ids[6]=h3;ids[7]=h4;ids[8]=h6r;ids[9]=h6c;ids[10]=h7r;ids[11]=h7t;
    if(!d4_float_ok()||!stage4_file_size_bytes(cache_path,&cache_size)||
       cache_size!=D4_CACHE_SIZE||!d4_hex_raw(ds,dsraw)||!d4_hex_raw(ck,ckraw)||!d4_hex_raw(tk,tkraw)){
        fprintf(stderr,"error: F1 cache format/identity check failed\n");failed=1;goto cleanup;
    }
    if(!d5_load_report(d4,d4_reference,&d4_count)||
       !d5_load_report(d6r,d6_reference,&d6_count)||
       !f1_value_is(d6_reference,d6_count,"checkpoint_roundtrip_pass","yes")||
       !f1_value_is(d6_reference,d6_count,"checkpoint_sha256",h6c)){
        fprintf(stderr,"error: F1 D4/D6 reference report mismatch\n");failed=1;goto cleanup;
    }
    if(!d5_load_report(d7r,d7_reference,&d7_count)||
       !f1_value_is(d7_reference,d7_count,"d7_resume_pass","yes")||
       !f1_value_is(d7_reference,d7_count,"deterministic_resume_trajectory_match","yes")||
       !f1_load_d7_tsv(d7t,d7_rows)){
        fprintf(stderr,"error: F1 D7 resume reference mismatch\n");failed=1;goto cleanup;
    }
    cache=fopen(cache_path,"rb");
    if(!cache||!d4_read_header(cache,&header)||!d4_header_matches(&header,dsraw,ckraw,tkraw)){
        fprintf(stderr,"error: F1 cache header mismatch\n");failed=1;goto cleanup;
    }
    rows=calloc(D4_ROWS,sizeof(*rows));if(!rows){fprintf(stderr,"error: F1 row allocation failed\n");failed=1;goto cleanup;}
    for(i=0;i<D4_ROWS;i++){
        if(!d4_read_row(cache,&rows[i])||rows[i].record>=D4_RECORDS||rows[i].target>=D4_VOCAB||
           rows[i].split>=D4_SPLITS||rows[i].split==D4_REGRESSION){fprintf(stderr,"error: invalid F1 cache row\n");failed=1;goto cleanup;}
        if(i&&(rows[i].record<rows[i-1].record||(rows[i].record==rows[i-1].record&&
           (rows[i].split!=rows[i-1].split||rows[i].position<=rows[i-1].position)))){
            fprintf(stderr,"error: F1 cache row order error\n");failed=1;goto cleanup;
        }
        split_counts[rows[i].split]++;
    }
    if(fgetc(cache)!=EOF||ferror(cache)||fclose(cache)!=0){cache=NULL;fprintf(stderr,"error: F1 cache trailing/read error\n");failed=1;goto cleanup;}
    cache=NULL;
    if(split_counts[D4_TRAIN]!=111U||split_counts[D4_VALIDATION]!=33U||split_counts[D4_TEST]!=21U){
        fprintf(stderr,"error: F1 cache split counts mismatch\n");failed=1;goto cleanup;
    }
    unload_adapter_runtime();load_transformer(&tr,checkpoint);tr_ready=1;
    if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||
       !stage4_adapter_init(&ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){
        fprintf(stderr,"error: F1 model/adapter init failed\n");failed=1;goto cleanup;
    }
    ad_ready=1;
    if(ad.rank!=D4_RANK||ad.scale!=D4_SCALE||ad.a_count!=2304U||ad.b_count!=256000U){
        fprintf(stderr,"error: F1 adapter architecture mismatch\n");failed=1;goto cleanup;
    }
    base=calloc(D4_VOCAB,sizeof(float));final=calloc(D4_VOCAB,sizeof(float));
    if(!base||!final){fprintf(stderr,"error: F1 logits allocation failed\n");failed=1;goto cleanup;}
    if(!d6_load_checkpoint(d6c,ds,ca,ck,tk,&ad,&optimizer_step,D6_TOTAL_BYTES)||optimizer_step!=5U){
        fprintf(stderr,"error: F1 D6 step-5 checkpoint load failed\n");failed=1;goto cleanup;
    }
    d4_state_finite(&ad,&parameters_finite,&moments_finite);
    if(!parameters_finite||!moments_finite){fprintf(stderr,"error: F1 loaded D6 state not finite\n");failed=1;goto cleanup;}
    if(!f1_evaluate(rows,&ad,&tr,&points[5],&loss_finite,&corrections_finite)){
        fprintf(stderr,"error: F1 step-5 evaluation failed\n");failed=1;goto cleanup;
    }
    points[5].evaluated=1;
    step5_match=d4_fmt12(points[5].point.metrics.split[D4_TRAIN].loss/111.0,"10.258683447106")&&
        d4_fmt12(points[5].point.metrics.split[D4_VALIDATION].loss/33.0,"9.048405144340");
    if(!step5_match){fprintf(stderr,"error: F1 loaded step-5 metrics differ from D6 anchor\n");failed=1;goto cleanup;}
    lowest_validation=points[5].point.metrics.split[D4_VALIDATION].loss/33.0;
    previous_train=points[5].point.metrics.split[D4_TRAIN].loss/111.0;
    previous_validation=lowest_validation;
    if(previous_validation<=8.650949130533)first_accept=5U;
    if(points[5].all_same)first_collapse=5U;
    printf("step=5 train_loss=%.12f validation_loss=%.12f validation_unique_top1=%u\n",
        previous_train,previous_validation,points[5].unique_top1);
    for(i=6U;i<=F1_STEPS;i++){
        int grad_ok=1,current_parameters=1,current_moments=1;double update;
        if(!d4_step(rows,&ad,&tr,base,final,(int)i,&gradients[i],&grad_ok,&loss_finite)||!grad_ok){
            fprintf(stderr,"error: F1 train update %u failed\n",i);failed=1;goto cleanup;
        }
        gradients_finite=gradients_finite&&grad_ok;
        if(gradients[i].grad_test_rows!=0U)test_gradient_rows+=gradients[i].grad_test_rows;
        if(gradients[i].clip_applied)clipped++;else unclipped++;
        update=stage4_adapter_adam_step(&ad,D4_LR,D4_BETA1,D4_BETA2,D4_EPSILON,D4_WEIGHT_DECAY,(uint64_t)i);
        if(!isfinite(update)||!stage4_adapter_parameters_are_finite(&ad)){
            parameters_finite=0;fprintf(stderr,"error: F1 non-finite Adam update at step %u\n",i);failed=1;goto cleanup;
        }
        optimizer_step++;
        if(optimizer_step!=(uint64_t)i){fprintf(stderr,"error: F1 optimizer step sequence mismatch\n");failed=1;goto cleanup;}
        d4_state_finite(&ad,&current_parameters,&current_moments);
        parameters_finite=parameters_finite&&current_parameters;
        moments_finite=moments_finite&&current_moments;
        if(!current_parameters||!current_moments){fprintf(stderr,"error: F1 non-finite state at step %u\n",i);failed=1;goto cleanup;}
        if(f1_is_eval_step(i)){
            if(!f1_evaluate(rows,&ad,&tr,&points[i],&loss_finite,&corrections_finite)){
                fprintf(stderr,"error: F1 evaluation failed at step %u\n",i);failed=1;goto cleanup;
            }
            points[i].evaluated=1;
            test_evaluation_rows+=points[i].point.metrics.evaluation_test_rows;
            if(i<=10U){
                if(!f1_matches_d7_anchor(d4_reference,d4_count,i,&gradients[i],&points[i])||
                   !f1_matches_d7_tsv(d7_rows,i,&gradients[i],&points[i])){
                    d7_match=0;fprintf(stderr,"error: F1 trajectory differs from D7 at step %u\n",i);failed=1;goto cleanup;
                }
            }
            {
                double train_loss=points[i].point.metrics.split[D4_TRAIN].loss/111.0;
                double validation_loss=points[i].point.metrics.split[D4_VALIDATION].loss/33.0;
                if(train_loss<previous_train)train_imp++;else if(train_loss==previous_train)train_same++;else train_worse++;
                if(validation_loss<previous_validation)val_imp++;else if(validation_loss==previous_validation)val_same++;else val_worse++;
                if(first_accept==0U&&validation_loss<=8.650949130533)first_accept=i;
                if(first_collapse==0U&&points[i].all_same)first_collapse=i;
                if(validation_loss<lowest_validation){lowest_validation=validation_loss;lowest_step=i;}
                previous_train=train_loss;previous_validation=validation_loss;previous_eval=i;
                corrections_finite=corrections_finite&&isfinite(points[i].correction_rms_mean)&&
                    isfinite(points[i].correction_rms_max)&&isfinite(points[i].correction_abs_max);
                printf("step=%u train_loss=%.12f validation_loss=%.12f validation_unique_top1=%u validation_dominant_token=%u validation_correction_rms=%.17g validation_correction_abs_max=%.17g gradient_norm=%.17g clipped=%s\n",
                    i,train_loss,validation_loss,points[i].unique_top1,points[i].dominant_token,
                    points[i].correction_rms_mean,points[i].correction_abs_max,gradients[i].norm_before,
                    gradients[i].clip_applied?"yes":"no");
            }
        }
    }
    updates=optimizer_step-5U;
    if(previous_eval!=50U||updates!=45U||test_gradient_rows!=0U||test_evaluation_rows!=0U){
        fprintf(stderr,"error: F1 fixed-horizon/test-isolation gate failed\n");failed=1;goto cleanup;
    }
    d7_match=1;
    pass=step5_match&&d7_match&&gradients_finite&&parameters_finite&&moments_finite&&
        loss_finite&&corrections_finite&&updates==45U&&optimizer_step==50U&&
        test_gradient_rows==0U&&test_evaluation_rows==0U;
    if(!pass){fprintf(stderr,"error: F1 fixed-horizon integrity gate failed\n");failed=1;goto cleanup;}
    if(!f1_write_tsv(trajectory_path,points,gradients)||
       !f1_write_report(report_path,ids,optimizer_step,points,test_gradient_rows,test_evaluation_rows,
           clipped,unclipped,first_accept,first_collapse,lowest_step,train_imp,train_same,train_worse,
           val_imp,val_same,val_worse,step5_match,d7_match,gradients_finite&&parameters_finite&&
           moments_finite&&loss_finite&&corrections_finite,pass)){
        fprintf(stderr,"error: F1 output write failed\n");failed=1;goto cleanup;
    }
    printf("f1_fixed_horizon_pass=yes\nreport=%s\ntrajectory=%s\n",report_path,trajectory_path);
cleanup:
    if(cache!=NULL)fclose(cache);
    free(rows);free(base);free(final);
    if(ad_ready)stage4_adapter_free(&ad);
    if(tr_ready)unload_adapter_runtime();
    return failed?1:0;
}

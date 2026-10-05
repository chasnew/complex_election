library(tidyverse)
library(cowplot)

ce_result_path <- '/Users/chanuwasaswamenakul/Documents/workspace/complex_election/img'
box_path <- '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'

lt_enep <- function(vote_props) {
  return(1/sum(vote_props^2))
}

gos_enep <- function(vote_props) {
  highest_prop <- max(vote_props)
  return(sum(vote_props/(vote_props+(highest_prop^2)-(vote_props^2))))
}

# (Tentative) Electoral results when start with 3 parties



# Homophily results with different history bias

party_num <- 2
votep_cols <- paste0('vote_prop', 0:(party_num))
seatp_cols <- paste0('seat_prop', 0:(party_num))

batch_id <- 2
pop_mag <- 5
sim_type <- 'geo_sort'
elect_system_list <- c('one_per_party', 'proportional_rep')
beta_list <- c(0.1, 0.5, 0.9)
new_party <- 'moderate'

# agg_result_list <- list()
isort_plot_list <- list()

for (i in 1:length(elect_system_list)) {
  
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', new_party,
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
  elect_results <- step_results[c(votep_cols, seatp_cols)]
  elect_results[is.na(elect_results)] <- 0
  step_results[c(votep_cols, seatp_cols)] <- elect_results
  
  step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
  
  last_step <- max(step_results$step)
  
  # iterate over different degrees of history bias (beta)
  for (j in 1:length(beta_list)) {
    
    beta_tmp <- beta_list[j]
    
    isort_agg_result <- step_results %>%
      filter(step == last_step, beta == beta_tmp)
    
    # isort_agg_result %>% 
    #   filter(ideo_sort == 0.4, alpha == 0.9) %>% 
    #   select(vote_prop0:vote_prop2, lt_enep)
    
    panel_ind <- (i-1)*length(beta_list) + j
    isort_plot_list[[panel_ind]] <- isort_agg_result %>% 
      ggplot(aes(x=ideo_sort, y=lt_enep, color=as.factor(alpha))) +
      geom_point(position = position_jitter(w = 0.03, h = 0), alpha=0.1) +
      scale_x_continuous(expression(paste("Homophily (", h, ")")), expand=c(0.02,0.02)) +
      scale_y_continuous("ENEP", limits=c(1,3.1)) +
      scale_color_manual(values = c("#7ad151", "#2a788e", "#440154"),
                         labels = c(0.1, 0.5, 0.9)) +
      guides(color=guide_legend(title=expression(paste("Strategic\nTendency (", alpha, ")")),
                                override.aes = list(alpha = 1))) +
      theme_classic() +
      theme(axis.title=element_text(size=14),
            axis.text=element_text(size=12),
            legend.title = element_text(size=14),
            legend.text = element_text(size=12))
    
  }
}

# agg_sort_results <- rbind(agg_result_list[[1]], agg_result_list[[2]])

# extract legend and remove it from the plot
isort_legend <- get_legend(isort_plot_list[[1]])

for (i in 1:length(isort_plot_list)) {
  isort_plot_list[[i]] <- isort_plot_list[[i]] + theme(legend.position = "none")
}


# labels
letters <- c("a", "b", "c", "d", "e", "f")
label_list <- list()
for (i in 1:6) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0.01, y = 0.15,
      hjust = 0, vjust = 0
    )
}

hist_bias_title <- ggdraw() + 
  draw_label(
    "History Bias",
    fontface = 'bold',
    size = 18,
    y = 0.5,
    hjust = 0.5,
    angle = 90
  )

beta_row_label <- list()
for (i in 1:length(beta_list)) {
  beta_row_label[[i]] <- ggdraw() + 
    draw_label(
      bquote(beta==.(beta_list[i])),
      fontface = 'bold',
      size = 16,
      y = 0.6,
      vjust = 0.5,
      angle = 90
    )
}

fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )


# combining 3 panels into one figure
sup_fig1_main <- plot_grid(NA, fptp_title, prop_title, NA,
                           NA, label_list[[1]], label_list[[2]], NA,
                           beta_row_label[[1]], isort_plot_list[[1]], isort_plot_list[[4]], NA,
                           NA, label_list[[3]], label_list[[4]], NA,
                           beta_row_label[[2]], isort_plot_list[[2]], isort_plot_list[[5]], isort_legend,
                           NA, label_list[[5]], label_list[[6]], NA,
                           beta_row_label[[3]], isort_plot_list[[3]], isort_plot_list[[6]], NA,
                           ncol = 4, rel_widths = c(0.1, 1, 1, 0.3),
                           rel_heights = c(0.15, 0.1, 1, 0.1, 1, 0.1, 1))

sup_fig1 <- plot_grid(hist_bias_title, sup_fig1_main,
                      ncol = 2, rel_widths = c(0.05, 1))

ggsave(file.path(ce_result_path, "election_supfig1.jpg"),
       plot = sup_fig1,
       height = 10, width = 10)











# Full parameter sweep of radicalization in many small panel figures 

sim_type <- 'radicalization'
# elect_system <- 'one_per_party'
party_num <- 2
pop_mag <- 5
new_party <- 'None'
batch_id <- 2

# sim_type_list <- c('radicalization', 'vote_capture')
elect_system_list <- c('one_per_party', 'proportional_rep')
sgmt_col_names <- c("elect_system", "alpha", "beta", "ppos_start", "vote_prop_start", "share_deviation_start",
                    "step_start", "ppos_end", "vote_prop_end", "share_deviation_end", "step_end")

all_rad_results <- data.frame()

for (i in 1:length(elect_system_list)) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', new_party,
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  all_rad_results <- rbind(all_rad_results, step_results)
}

# extract the final vote proportion of p1's current position and calculate vote share deviation
agg_rad_results <- all_rad_results %>%
  group_by(elect_system, sim_id, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = last(vote_prop0),
            step = last(step)) %>% 
  mutate(prop_share = (((mov_ppos + 0.5)/2) + (1 - mov_ppos))/2,
         share_deviation = vote_prop0 - prop_share) %>%  
  arrange(sim_id, step)

# average over all simulations to get the mean trajectory
avg_rad_results <- agg_rad_results %>% 
  group_by(elect_system, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = mean(vote_prop0),
            share_deviation = mean(share_deviation),
            step = last(step))

# FPTP
avg_fptp_results <- avg_rad_results %>% 
  filter(elect_system == "one_per_party") %>% 
  arrange(step)

# unique combinations of alpha x beta = 121
unique_ab_num <- avg_fptp_results %>%
  distinct(alpha, beta) %>% 
  nrow()

avg_fptp_sgmts <- cbind(avg_fptp_results[-((nrow(avg_fptp_results)-(unique_ab_num-1)):nrow(avg_fptp_results)),],
                        avg_fptp_results[-(1:unique_ab_num),4:7])
colnames(avg_fptp_sgmts) <- sgmt_col_names

# PR
avg_pr_results <- avg_rad_results %>% 
  filter(elect_system == "proportional_rep") %>% 
  arrange(step)

avg_pr_sgmts <- cbind(avg_pr_results[-((nrow(avg_pr_results)-(unique_ab_num-1)):nrow(avg_pr_results)),],
                      avg_pr_results[-(1:unique_ab_num),4:7])
colnames(avg_pr_sgmts) <- sgmt_col_names

avg_segments <- rbind(avg_fptp_sgmts, avg_pr_sgmts)

# subset rows from segments df using arrow_steps to create df to draw the arrows
arrow_steps <- seq(9, 49, by=30)

# arrow segments
arrow_segments <- avg_segments %>% 
  filter(step_start %in% arrow_steps)

fptp_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "one_per_party")

pr_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "proportional_rep")

# normal line segments
avg_segments <- avg_segments %>% 
  filter(!(step_start %in% arrow_steps))

fptp_avg_segments <- avg_segments %>% 
  filter(elect_system == "one_per_party")

pr_avg_segments <- avg_segments %>% 
  filter(elect_system == "proportional_rep")

# Creating (multiple) plots

# Vote share over time
# FPTP
fptp_voteshare <- agg_rad_results %>% 
  filter(elect_system == 'one_per_party') %>% 
  ggplot(aes(x=step, y=vote_prop0, group=sim_id)) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=fptp_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=fptp_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow(length = unit(0.25,"cm"))) +
  facet_grid(beta ~ alpha) +
  geom_vline(xintercept = 25, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous("Vote Share", limits = c(0,1)) +
  theme_bw() +
  theme(axis.title=element_text(size=18),
        strip.text=element_text(size=12, face="bold"),
        axis.text=element_blank(),
        axis.ticks=element_blank())

# PR
pr_voteshare <- agg_rad_results %>% 
  filter(elect_system == 'proportional_rep') %>% 
  ggplot(aes(x=step, y=vote_prop0, group=sim_id)) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=pr_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=pr_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow(length = unit(0.25,"cm"))) +
  facet_grid(beta ~ alpha) +
  geom_vline(xintercept = 25, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous("Vote Share", limits = c(0,1)) +
  theme_bw() +
  theme(axis.title=element_text(size=18),
        strip.text=element_text(size=12, face="bold"),
        axis.text=element_blank(),
        axis.ticks=element_blank())


fptp_title <- ggdraw() +
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 18,
    y = 0.5,
    hjust = 0.5,
    angle = 90
  )

prop_title <- ggdraw() +
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 18,
    y = 0.5,
    hjust = 0.5,
    angle = 90
  )

strat_title <- ggdraw() +
  draw_label(
    "Strategic Tendency",
    fontface = 'bold',
    size = 16,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

hist_bias_title <- ggdraw() + 
  draw_label(
    "History Bias",
    fontface = 'bold',
    size = 16,
    y = 0.5,
    hjust = 0.5,
    angle = 270
  )

# ggsave(file.path(ce_result_path, "election_supfig2.jpg"),
#        plot = sup_fig2,
#        height = 8, width = 12)


sup_fig2_main <- plot_grid(NA, strat_title,
                           fptp_title, fptp_voteshare,
                           prop_title, pr_voteshare,
                           ncol=2, rel_widths=c(0.05, 1),
                           rel_heights=c(0.05, 1, 1))

sup_fig2 <- plot_grid(sup_fig2_main, hist_bias_title,
                      ncol=2, rel_widths=c(1, 0.025))

ggsave(file.path(ce_result_path, "election_supfig2.jpg"),
       plot = sup_fig2,
       height = 12, width = 12)








# Radicalized party against emerged party

sim_type <- 'radicalization_and_emerge'
# elect_system <- 'one_per_party'
party_num <- 2
pop_mag <- 5
new_party <- 'moderate'
batch_id <- 2

# sim_type_list <- c('radicalization', 'vote_capture')
elect_system_list <- c('one_per_party', 'proportional_rep')
sgmt_col_names <- c("elect_system", "alpha", "beta", "ppos_start", "vote_prop_start", "share_deviation_start",
                    "step_start", "ppos_end", "vote_prop_end", "share_deviation_end", "step_end")

all_rad_results <- data.frame()

for (i in 1:length(elect_system_list)) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', new_party,
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  all_rad_results <- rbind(all_rad_results, step_results)
}

# keep only rows after party 3 (new moderate party entered the electoral race)
post_entry_rad <- all_rad_results %>% 
  filter(!is.na(vote_prop2))

# extract the final vote proportion of p1's current position and calculate vote share deviation
agg_rad_emerge_results <- post_entry_rad %>%
  group_by(elect_system, sim_id, alpha, beta) %>% 
  summarize(vote_prop2 = mean(vote_prop2), mov_ppos = last(mov_ppos)) %>% 
  mutate(prop_share = ((0.5/2) + (0.6/2))/2,
         share_deviation = vote_prop2 - prop_share) %>%  
  arrange(sim_id)

# average over all simulations to get the mean trajectory
avg_rad_emerge_results <- agg_rad_emerge_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6)) %>% 
  group_by(elect_system, alpha, beta) %>% 
  summarize(std_err = sd(vote_prop2)/sqrt(n()),
            vote_prop2 = mean(vote_prop2),
            deviation_serr = sd(share_deviation)/sqrt(n()),
            share_deviation = mean(share_deviation))


# Creating (multiple) plots
# Ideology over time

ideo_change <- all_rad_results %>% 
  distinct(mov_ppos, step) %>% 
  ggplot(aes(x=step, y=mov_ppos)) + 
  geom_line() +
  annotate("segment", y=0,yend=0.5,x=12.5,xend=12.5, linewidth=1, linetype="dashed") +
  annotate("segment", y=0.5,yend=0.5,x=0,xend=12.5, linewidth=1, linetype="dashed") +
  annotate("segment", y=0,yend=0.6,x=15,xend=15, linewidth=1, linetype="dashed") +
  annotate("segment", y=0.6,yend=0.6,x=0,xend=15, linewidth=1, linetype="dashed") +
  annotate("text", y=0.4,x=5, label="Radicalization\nThreshold", size=5) +
  annotate("text", y=0.65,x=8, label="Moderate 3rd-Party Entry", size=5) +
  scale_x_continuous("Electoral Cycle", expand = c(0.01,0.01)) +
  scale_y_continuous("Party Ideology", limits = c(0,1), expand = c(0.01,0.01)) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14))


# Vote share of a new moderate party
newp_vote_share <- avg_rad_emerge_results %>% 
  ggplot(aes(x=elect_system, y=vote_prop2, color=interaction(alpha, beta))) +
  geom_point(size=2.5, position=position_dodge(width=0.5)) +
  geom_hline(yintercept = ((0.5/2) + (0.6/2))/2, linetype = "dashed") +
  geom_errorbar(aes(ymin=vote_prop2-(2*std_err),
                    ymax=vote_prop2+(2*std_err)),
                width=.25, position=position_dodge(width=0.5)) +
  annotate("text", y=0.285,x=1, label=expression(paste(s[sincere])), size=6) +
  scale_x_discrete("Electoral System", labels = c("First-Past-The-Post", "Proportional\nRepresentation")) +
  scale_y_continuous(expression(paste("Vote Share (", s[3], ")"))) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_manual(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.3)),
                                expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.6)),
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6))),
                     values = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")) +
  theme_classic() +
  theme(axis.title.x=element_blank(),
        axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.title = element_text(size=16),
        legend.text = element_text(size=14))


letters <- c("a", "b")
label_list <- list()
for (i in 1:2) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 18,
      x = 0, y = 0.1,
      hjust = 0, vjust = 0
    )
}

supfig3 <- plot_grid(label_list[[1]], NA, label_list[[2]],
                     ideo_change, NA, newp_vote_share,
                     ncol = 3, rel_widths = c(0.7, 0.05, 1),
                     rel_heights = c(0.1, 1))

ggsave(file.path(ce_result_path, "election_supfig3.jpg"),
       plot = supfig3, height = 6, width = 12)









# Realistic setting w/ 650 seats in total (50 seats per district in PR)

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 6
pop_mag <- 125
op_distr <- 'uniform'
sim_type <- 'when_pemerge'

elect_system_list <- c('one_per_party', 'proportional_rep')

all_rad_results <- data.frame()
heatmap_list <- list()

party_num <- 2

for (i in 1:length(elect_system_list)) {
  # radicalization result
  rad_result_path <- file.path(box_path, 'ComplexElection', 'results',
                               paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                      '_p', party_num, '_None_radicalization', batch_id, '.csv'))
  
  step_results <- read_csv(rad_result_path) %>%
    select(-district_num, -rep_num)
  
  all_rad_results <- rbind(all_rad_results, step_results)
  
  
  # extreme party emergence result
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', 'extreme',
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
  elect_results <- step_results[c(votep_cols, seatp_cols)]
  elect_results[is.na(elect_results)] <- 0
  step_results[c(votep_cols, seatp_cols)] <- elect_results
  
  step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
  
  last_step <- max(step_results$step)
  stable_results <- step_results %>% 
    filter(step > 5, alpha < 1) %>% 
    group_by(sim_id, alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep)) %>% 
    group_by(alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep))
  
  # Original palette: low="#FFEDBB", mid="#FEB24C", high="#FF3F21"
  # Viridis: low="#440154", mid="#21918c", high="#fde725"
  # New palette: low="#009E73", mid="#DDFFDD", high="#CC79A7"
  enep_heatmap <- stable_results %>% 
    ggplot(aes(x=alpha, y=beta, fill=lt_enep)) +
    geom_raster() +
    scale_fill_gradient2(name="ENEP", midpoint=1.9,
                         low="#009E73", mid="#DDFFDD", high="#CC79A7",
                         limits = c(1, 2.7)) +
    # scale_fill_distiller(palette = "YlOrRd", limits=c(1,3)) +
    scale_x_continuous(expression(paste("Strategic Tendency (", alpha, ")")), expand=c(0,0)) +
    scale_y_continuous(expression(paste("History Bias (", beta, ")")), expand=c(0,0)) +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          legend.title = element_text(size=14),
          legend.text = element_text(size=12))
  
  panel_ind <- i
  heatmap_list[[panel_ind]] <- enep_heatmap
}

# extract shared legend and remove legends from both plots
heatmap_legend <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")


# radicalization result processing
sgmt_col_names <- c("elect_system", "alpha", "beta", "ppos_start", "vote_prop_start", "share_deviation_start",
                    "step_start", "ppos_end", "vote_prop_end", "share_deviation_end", "step_end")


# extract the final vote proportion of p1's current position and calculate vote share deviation
agg_rad_results <- all_rad_results %>%
  group_by(elect_system, sim_id, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = last(vote_prop0),
            step = last(step)) %>% 
  mutate(prop_share = (((mov_ppos + 0.5)/2) + (1 - mov_ppos))/2,
         share_deviation = vote_prop0 - prop_share) %>%  
  arrange(sim_id, step)

# average over all simulations to get the mean trajectory
avg_rad_results <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6)) %>% 
  group_by(elect_system, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = mean(vote_prop0),
            share_deviation = mean(share_deviation),
            step = last(step))

# FPTP
avg_fptp_results <- avg_rad_results %>% 
  filter(elect_system == "one_per_party") %>% 
  arrange(step)

avg_fptp_sgmts <- cbind(avg_fptp_results[-((nrow(avg_fptp_results)-3):nrow(avg_fptp_results)),],
                        avg_fptp_results[-(1:4),4:7])
colnames(avg_fptp_sgmts) <- sgmt_col_names

# PR
avg_pr_results <- avg_rad_results %>% 
  filter(elect_system == "proportional_rep") %>% 
  arrange(step)

avg_pr_sgmts <- cbind(avg_pr_results[-((nrow(avg_pr_results)-3):nrow(avg_pr_results)),],
                      avg_pr_results[-(1:4),4:7])
colnames(avg_pr_sgmts) <- sgmt_col_names

avg_segments <- rbind(avg_fptp_sgmts, avg_pr_sgmts)

# subset rows from segments df using arrow_steps to create df to draw the arrows
arrow_steps <- seq(9, 49, by=10)

# arrow segments
arrow_segments <- avg_segments %>% 
  filter(step_start %in% arrow_steps)

fptp_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "one_per_party")

pr_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "proportional_rep")

# normal line segments
avg_segments <- avg_segments %>% 
  filter(!(step_start %in% arrow_steps))

fptp_avg_segments <- avg_segments %>% 
  filter(elect_system == "one_per_party")

pr_avg_segments <- avg_segments %>% 
  filter(elect_system == "proportional_rep")

# Vote share over time
# FPTP
fptp_voteshare <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'one_per_party') %>% 
  ggplot(aes(x=step, y=vote_prop0, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=fptp_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=fptp_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste("Vote Share (", s[actual], ")")), limits = c(0,1)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_manual(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.3)),
                                expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.6)),
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6))),
                     values = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.title = element_text(size=16),
        legend.text = element_text(size=14))

# PR
pr_voteshare <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'proportional_rep') %>% 
  ggplot(aes(x=step, y=vote_prop0, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=pr_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=pr_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste("Vote Share (", s[actual], ")")), limits = c(0,1)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_discrete(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)),
                                  expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)),
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)))) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.position = "None")

# extract shared legend and remove legends from both plots
shared_rad_legend <- get_legend(fptp_voteshare)
fptp_voteshare <- fptp_voteshare + theme(legend.position = "none")


# FPTP and PR title labels
fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

letters <- c("a", "b", "c", "d")
label_list <- list()
for (i in 1:4) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0, y = 0.1,
      hjust = 0, vjust = 0
    )
}

supfig7 <- plot_grid(fptp_title, prop_title, NA,
                     label_list[[1]], label_list[[2]], NA,
                     heatmap_list[[1]], heatmap_list[[2]], heatmap_legend,
                     label_list[[3]], label_list[[4]], NA,
                     fptp_voteshare, pr_voteshare, shared_rad_legend,
                     ncol = 3, rel_widths = c(1, 1, 0.25), rel_heights = c(0.1, 0.1, 1, 0.1, 1))

ggsave(file.path(ce_result_path, "election_supfig7.jpg"),
       plot = supfig7, height = 8, width = 14)













# LT vs Gosolov measure of ENEP

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 2
pop_mag <- 5
op_distr <- 'uniform'
sim_type <- 'when_pemerge'
new_party_list <- c('moderate', 'extreme')

elect_system_list <- c('one_per_party', 'proportional_rep')
heatmap_list <- list()

party_num <- 2

for (i in 1:length(new_party_list)) {
  new_party <- new_party_list[i]
  
  for (j in 1:length(elect_system_list)) {
    # moderate party emergence result
    result_path <- file.path(box_path, 'ComplexElection', 'results',
                             paste0('estep', pop_mag, 'k_', elect_system_list[j],
                                    '_p', party_num, '_', new_party,
                                    '_', sim_type, batch_id, '.csv'))
    
    step_results <- read_csv(result_path) %>%
      select(-district_num, -rep_num)
    
    # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
    elect_results <- step_results[c(votep_cols, seatp_cols)]
    elect_results[is.na(elect_results)] <- 0
    step_results[c(votep_cols, seatp_cols)] <- elect_results
    
    step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
    step_results$gos_enep <- apply(step_results[votep_cols], 1, gos_enep)
    
    last_step <- max(step_results$step)
    stable_results <- step_results %>% 
      filter(step > 5, alpha < 1) %>% 
      group_by(sim_id, alpha, beta) %>% 
      summarize(lt_enep = mean(lt_enep), gos_enep = mean(gos_enep)) %>% 
      group_by(alpha, beta) %>% 
      summarize(lt_enep = mean(lt_enep), gos_enep = mean(gos_enep))
    
    # stable_results %>% 
    #   group_by(alpha, beta) %>% 
    #   summarize(vote_prop2.avg = mean(vote_prop2), lt_enep.avg = mean(lt_enep), lt_enep.sd = sd(lt_enep)) %>% 
    #   filter(alpha > 0.5, beta > 0.5)
    
    # stable_results %>% 
    #   group_by(sim_id, alpha, beta) %>% 
    #   summarize(lt_enep = mean(lt_enep)) %>% 
    #   filter(alpha == 0.9, beta == 0.6, lt_enep < 2.5)
    # 
    # stable_results %>%
    #   filter(alpha == 0.9, beta == 0.6, sim_id == 0)
    # 
    # stable_results %>% 
    #   filter(alpha == 0.9, beta == 0.6) %>% 
    #   ggplot(aes(x=step, y=lt_enep, color=as.factor(sim_id), group=sim_id)) +
    #   geom_line()
    
    # Original palette: low="#FFEDBB", mid="#FEB24C", high="#FF3F21"
    # Viridis: low="#440154", mid="#21918c", high="#fde725"
    # New palette: low="#009E73", mid="#DDFFDD", high="#CC79A7"
    enep_heatmap <- stable_results %>% 
      ggplot(aes(x=alpha, y=beta, fill=lt_enep)) +
      geom_raster() +
      scale_fill_gradient2(name="ENEP", midpoint=2,
                           low="#009E73", mid="#DDFFDD", high="#CC79A7",
                           limits = c(1, 3)) +
      # scale_fill_distiller(palette = "YlOrRd", limits=c(1,3)) +
      scale_x_continuous(expression(paste("Strategic Tendency (", alpha, ")")), expand=c(0,0)) +
      scale_y_continuous(expression(paste("History Bias (", beta, ")")), expand=c(0,0)) +
      theme(axis.title=element_text(size=14),
            axis.text=element_text(size=12),
            legend.title = element_text(size=14),
            legend.text = element_text(size=12))
    
    panel_ind <- 2*(i-1) + j
    heatmap_list[[panel_ind]] <- enep_heatmap
  }
}

# extract shared legend and remove legends from both plots
heatmap_legend <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")
heatmap_list[[3]] <- heatmap_list[[3]] + theme(legend.position = "none")
heatmap_list[[4]] <- heatmap_list[[4]] + theme(legend.position = "none")


fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )


mnew_title <- ggdraw() + 
  draw_label(
    "New Moderate Party",
    fontface = 'bold',
    size = 16,
    y = 0.55,
    hjust = 0.5,
    angle = 90
  )

enew_title <- ggdraw() + 
  draw_label(
    "New Extreme Party",
    fontface = 'bold',
    size = 16,
    y = 0.55,
    hjust = 0.5,
    angle = 90
  )

main_grid5 <- plot_grid(NA, fptp_title, prop_title,
                        mnew_title, heatmap_list[[1]], heatmap_list[[2]],
                        enew_title, heatmap_list[[3]], heatmap_list[[4]],
                        ncol = 3, labels = c(NA, NA, NA,
                                             NA, "a", "b",
                                             NA, "c", "d"), label_size = 16,
                        rel_widths = c(0.1, 1, 1), rel_heights = c(0.1, 1, 1))

supfig4 <- plot_grid(main_grid5, heatmap_legend,
                     ncol = 2, rel_widths = c(1, 0.1))

ggsave(file.path(ce_result_path, "election_supfig4.jpg"),
       plot = supfig4, height = 8, width = 14)








# D'Hondt vs Hamilton seat allocation method

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 5
pop_mag <- 5
op_distr <- 'uniform'
sim_type <- 'when_pemerge'

elect_system <- 'proportional_rep'
new_party_list <- c('moderate', 'extreme')
heatmap_list <- list()

party_num <- 2

for (i in 1:length(new_party_list)) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system,
                                  '_p', party_num, '_', new_party_list[i],
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
  elect_results <- step_results[c(votep_cols, seatp_cols)]
  elect_results[is.na(elect_results)] <- 0
  step_results[c(votep_cols, seatp_cols)] <- elect_results
  
  step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
  
  last_step <- max(step_results$step)
  stable_results <- step_results %>% 
    filter(step > 5, alpha < 1) %>% 
    group_by(sim_id, alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep)) %>% 
    group_by(alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep))
  
  # stable_results %>% 
  #   group_by(alpha, beta) %>% 
  #   summarize(vote_prop2.avg = mean(vote_prop2), lt_enep.avg = mean(lt_enep), lt_enep.sd = sd(lt_enep)) %>% 
  #   filter(alpha > 0.5, beta > 0.5)
  
  # stable_results %>% 
  #   group_by(sim_id, alpha, beta) %>% 
  #   summarize(lt_enep = mean(lt_enep)) %>% 
  #   filter(alpha == 0.9, beta == 0.6, lt_enep < 2.5)
  # 
  # stable_results %>%
  #   filter(alpha == 0.9, beta == 0.6, sim_id == 0)
  # 
  # stable_results %>% 
  #   filter(alpha == 0.9, beta == 0.6) %>% 
  #   ggplot(aes(x=step, y=lt_enep, color=as.factor(sim_id), group=sim_id)) +
  #   geom_line()
  
  # Original palette: low="#FFEDBB", mid="#FEB24C", high="#FF3F21"
  # Viridis: low="#440154", mid="#21918c", high="#fde725"
  # New palette: low="#009E73", mid="#DDFFDD", high="#CC79A7"
  enep_heatmap <- stable_results %>% 
    ggplot(aes(x=alpha, y=beta, fill=lt_enep)) +
    geom_raster() +
    scale_fill_gradient2(name="ENEP", midpoint=2,
                         low="#009E73", mid="#DDFFDD", high="#CC79A7",
                         limits = c(1, 3)) +
    # scale_fill_distiller(palette = "YlOrRd", limits=c(1,3)) +
    scale_x_continuous(expression(paste("Strategic Tendency (", alpha, ")")), expand=c(0,0)) +
    scale_y_continuous(expression(paste("History Bias (", beta, ")")), expand=c(0,0)) +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          legend.title = element_text(size=14),
          legend.text = element_text(size=12))
  
  panel_ind <- i
  heatmap_list[[panel_ind]] <- enep_heatmap
}

# extract shared legend and remove legends from both plots
heatmap_legend <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")

mnew_title <- ggdraw() + 
  draw_label(
    "New Moderate Party",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

enew_title <- ggdraw() + 
  draw_label(
    "New Extreme Party",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

supfig5 <- plot_grid(mnew_title, enew_title, NA,
                     heatmap_list[[1]], heatmap_list[[2]], heatmap_legend,
                     ncol = 3, labels = c(NA, NA, NA,
                                          "a", "b", NA), label_size = 16,
                     rel_widths = c(1, 1, 0.2), rel_heights = c(0.1, 1))


ggsave(file.path(ce_result_path, "election_supfig5.jpg"),
       plot = supfig5, height = 4, width = 12)









# Droop quota results

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 7
pop_mag <- 5
op_distr <- 'uniform'
sim_type <- 'when_pemerge'

elect_system_list <- c('one_per_party', 'proportional_rep')

all_rad_results <- data.frame()
heatmap_list <- list()

party_num <- 2

for (i in 1:length(elect_system_list)) {
  # radicalization result
  rad_result_path <- file.path(box_path, 'ComplexElection', 'results',
                               paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                      '_p', party_num, '_None_radicalization', batch_id, '.csv'))
  
  step_results <- read_csv(rad_result_path) %>%
    select(-district_num, -rep_num)
  
  all_rad_results <- rbind(all_rad_results, step_results)
  
  
  # extreme party emergence result
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', 'extreme',
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
  elect_results <- step_results[c(votep_cols, seatp_cols)]
  elect_results[is.na(elect_results)] <- 0
  step_results[c(votep_cols, seatp_cols)] <- elect_results
  
  step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
  
  last_step <- max(step_results$step)
  stable_results <- step_results %>% 
    filter(step > 5, alpha < 1) %>% 
    group_by(sim_id, alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep)) %>% 
    group_by(alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep))
  
  # Original palette: low="#FFEDBB", mid="#FEB24C", high="#FF3F21"
  # Viridis: low="#440154", mid="#21918c", high="#fde725"
  # New palette: low="#009E73", mid="#DDFFDD", high="#CC79A7"
  enep_heatmap <- stable_results %>% 
    ggplot(aes(x=alpha, y=beta, fill=lt_enep)) +
    geom_raster() +
    scale_fill_gradient2(name="ENEP", midpoint=1.9,
                         low="#009E73", mid="#DDFFDD", high="#CC79A7",
                         limits = c(1, 2.7)) +
    # scale_fill_distiller(palette = "YlOrRd", limits=c(1,3)) +
    scale_x_continuous(expression(paste("Strategic Tendency (", alpha, ")")), expand=c(0,0)) +
    scale_y_continuous(expression(paste("History Bias (", beta, ")")), expand=c(0,0)) +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          legend.title = element_text(size=14),
          legend.text = element_text(size=12))
  
  panel_ind <- i
  heatmap_list[[panel_ind]] <- enep_heatmap
}

# extract shared legend and remove legends from both plots
heatmap_legend <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")


# radicalization result processing
sgmt_col_names <- c("elect_system", "alpha", "beta", "ppos_start", "vote_prop_start", "share_deviation_start",
                    "step_start", "ppos_end", "vote_prop_end", "share_deviation_end", "step_end")


# extract the final vote proportion of p1's current position and calculate vote share deviation
agg_rad_results <- all_rad_results %>%
  group_by(elect_system, sim_id, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = last(vote_prop0),
            step = last(step)) %>% 
  mutate(prop_share = (((mov_ppos + 0.5)/2) + (1 - mov_ppos))/2,
         share_deviation = vote_prop0 - prop_share) %>%  
  arrange(sim_id, step)

# average over all simulations to get the mean trajectory
avg_rad_results <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6)) %>% 
  group_by(elect_system, alpha, beta, mov_ppos) %>% 
  summarize(vote_prop0 = mean(vote_prop0),
            share_deviation = mean(share_deviation),
            step = last(step))

# FPTP
avg_fptp_results <- avg_rad_results %>% 
  filter(elect_system == "one_per_party") %>% 
  arrange(step)

avg_fptp_sgmts <- cbind(avg_fptp_results[-((nrow(avg_fptp_results)-3):nrow(avg_fptp_results)),],
                        avg_fptp_results[-(1:4),4:7])
colnames(avg_fptp_sgmts) <- sgmt_col_names

# PR
avg_pr_results <- avg_rad_results %>% 
  filter(elect_system == "proportional_rep") %>% 
  arrange(step)

avg_pr_sgmts <- cbind(avg_pr_results[-((nrow(avg_pr_results)-3):nrow(avg_pr_results)),],
                      avg_pr_results[-(1:4),4:7])
colnames(avg_pr_sgmts) <- sgmt_col_names

avg_segments <- rbind(avg_fptp_sgmts, avg_pr_sgmts)

# subset rows from segments df using arrow_steps to create df to draw the arrows
arrow_steps <- seq(9, 49, by=10)

# arrow segments
arrow_segments <- avg_segments %>% 
  filter(step_start %in% arrow_steps)

fptp_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "one_per_party")

pr_arrow_segments <- arrow_segments %>% 
  filter(elect_system == "proportional_rep")

# normal line segments
avg_segments <- avg_segments %>% 
  filter(!(step_start %in% arrow_steps))

fptp_avg_segments <- avg_segments %>% 
  filter(elect_system == "one_per_party")

pr_avg_segments <- avg_segments %>% 
  filter(elect_system == "proportional_rep")

# Vote share over time
# FPTP
fptp_voteshare <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'one_per_party') %>% 
  ggplot(aes(x=step, y=vote_prop0, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=fptp_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=fptp_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste("Vote Share (", s[actual], ")")), limits = c(0,1)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_manual(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.3)),
                                expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.6)),
                                expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6))),
                     values = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.title = element_text(size=16),
        legend.text = element_text(size=14))

# PR
pr_voteshare <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'proportional_rep') %>% 
  ggplot(aes(x=step, y=vote_prop0, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=pr_avg_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=pr_arrow_segments,
               aes(x=step_start, y=vote_prop_start,
                   xend=step_end, yend=vote_prop_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste("Vote Share (", s[actual], ")")), limits = c(0,1)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_discrete(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)),
                                  expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)),
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)))) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.position = "None")

# extract shared legend and remove legends from both plots
shared_rad_legend <- get_legend(fptp_voteshare)
fptp_voteshare <- fptp_voteshare + theme(legend.position = "none")


# FPTP and PR title labels
fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.55,
    vjust = 0.5,
    angle = 0
  )

letters <- c("a", "b", "c", "d")
label_list <- list()
for (i in 1:4) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0, y = 0.1,
      hjust = 0, vjust = 0
    )
}

supfig6 <- plot_grid(fptp_title, prop_title, NA,
                     label_list[[1]], label_list[[2]], NA,
                     heatmap_list[[1]], heatmap_list[[2]], heatmap_legend,
                     label_list[[3]], label_list[[4]], NA,
                     fptp_voteshare, pr_voteshare, shared_rad_legend,
                     ncol = 3, rel_widths = c(1, 1, 0.25), rel_heights = c(0.1, 0.1, 1, 0.1, 1))

ggsave(file.path(ce_result_path, "election_supfig6.jpg"),
       plot = supfig6, height = 8, width = 14)




# (Tentative) Vote capture scenario


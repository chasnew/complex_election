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


# Figure 0: Model and Simulation diagrams

model_diagram_file <- file.path(ce_result_path, "emodel_diagram3.jpg")
model_diagram <- ggdraw() +
  draw_image(
    model_diagram_file, scale = 1
  )

sort_diagram_file <- file.path(ce_result_path, "ideosort_diagram.jpg")
sort_diagram <- ggdraw() +
  draw_image(
    sort_diagram_file, scale = 1
  )

fig1_alabel <- ggdraw() + 
  draw_label(
    "a",
    fontface = 'bold',
    size = 18,
    x = 0, y = 0.1,
    hjust = 0, vjust = 0
  )

fig1_blabel <- ggdraw() + 
  draw_label(
    "b",
    fontface = 'bold',
    size = 18,
    x = 0, y = 0.1,
    hjust = 0, vjust = 0
  )

figure0 <- plot_grid(
  fig1_alabel, NA, fig1_blabel,
  model_diagram, NA, sort_diagram,
  ncol = 3, rel_widths = c(1, 0.05, 0.4),
  rel_heights = c(0.04, 1)
)

ggsave(file.path(ce_result_path, "election_fig0.jpg"), plot = figure0, width = 12, height = 5)






# Figure 1: Baseline electoral outcomes
# Distribution of ideology of representatives (fptp vs pr)
elect_system_list <- c('one_per_party', 'proportional_rep')
res_v_rep_plots <- list()
alpha_list <- c(1, 1.5)

all_elected_reps <- NULL

for (i in 1:length(elect_system_list)) {
  for (j in 1:length(alpha_list)) {
    elected_path <- file.path(box_path, 'ComplexElection', 'results',
                              paste0('elected_pref_', elect_system_list[i], '_',
                                     'early', alpha_list[j], '.csv'))
    print(elected_path)
    
    # resident_opis <- read_csv(resident_path, col_names = FALSE)
    elected_reps <- read_csv(elected_path) %>% 
      mutate(elect_system = elect_system_list[i],
             alpha = alpha_list[j])
    
    all_elected_reps <- rbind(all_elected_reps, elected_reps)
  }
}

balanced_reps_plot <- all_elected_reps %>% 
  filter(alpha == 1) %>% 
  ggplot(aes(x=pos, fill=elect_system)) +
  # geom_histogram(bins=100, alpha=0.5, position="identity") +
  geom_histogram(aes(y=0.05*..density..),
                 binwidth=0.05, alpha=0.5,
                 position='identity') +
  scale_x_continuous("Ideology", limits = c(-1,1)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(y = "Proportion", fill = NULL) +
  scale_fill_manual(values=c("#F8766D", "#619CFF"),
                    labels=c("First-Past-The-Post", "Proportional Representation")) +
  theme_classic() +
  theme(axis.title=element_text(size=16),
        axis.text=element_text(size=14),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(size=12),
        legend.direction = "horizontal")

biased_reps_plot <- all_elected_reps %>% 
  filter(alpha == 1.5) %>% 
  ggplot(aes(x=pos, fill=elect_system)) +
  # geom_histogram(bins=100, alpha=0.5, position="identity") +
  geom_histogram(aes(y=0.05*..density..),
                 binwidth=0.05, alpha=0.5,
                 position='identity') +
  scale_x_continuous("Ideology", limits = c(-1,1)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(y = "Proportion", fill = NULL) +
  scale_fill_manual(values=c("#F8766D", "#619CFF"),
                    labels=c("First-Past-The-Post", "Proportional Representation")) +
  theme_classic() +
  theme(axis.title=element_text(size=16),
        axis.text=element_text(size=14),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position="None")

# extract legend and remove it from the plot
tmp_legend <- get_legend(balanced_reps_plot)
balanced_reps_plot <- balanced_reps_plot + theme(legend.position = "None")

# plot_grid(balanced_reps_plot, biased_reps_plot, ncol = 1)
# padded_elect_avg <- plot_grid(NA, elect_avg_plot, ncol = 2, 
#                               rel_widths = c(0.02, 1))
# base_rep_grid <- plot_grid(padded_elect_avg, base_reps_plot,
#                            ncol = 1, rel_heights = c(0.07, 1))
# base_rep_grid


# Average representation in baseline scenario between fptp and pr
elect_system_list <- c('one_per_party', 'proportional_rep')
alpha_list <- c(1, 1.5)

represt_df <- NULL

for(elect_system in elect_system_list) {
  for(alpha in alpha_list) {
    result_path <- file.path(box_path, 'ComplexElection', 'results',
                             paste0('estep5k_', elect_system, '_p2_None_baseline', alpha, '.csv'))
    
    # load in representativeness results and aggregate by simulation run
    represt_results <- read_csv(result_path) %>%
      select(elect_system, sim_id, avg_close_elected) %>% 
      group_by(elect_system, sim_id) %>% 
      summarize(avg_dist = last(avg_close_elected)) %>% 
      mutate(alpha = alpha)
    
    # calculate representativeness for each simulation with a given set of parameters (alpha, beta)
    # average distant range [0, 2]
    represt_tmp <- represt_results %>% 
      mutate(inv_dist = 1 - (avg_dist/2))
    
    represt_df <- rbind(represt_df, represt_tmp)
  }
}

# summarize representation data to make dot and whisker in the plot
represt_summary <- represt_df %>% 
  group_by(elect_system, alpha) %>% 
  summarize(sd_reprs = sd(inv_dist), inv_dist = mean(inv_dist),
            data_count = n()) %>% 
  mutate(se_reprs = sd_reprs/sqrt(data_count))


balanced_reprst_plot <- represt_df %>% 
  filter(alpha == 1) %>% 
  ggplot(aes(x=elect_system, y=inv_dist, fill=elect_system)) +
  geom_point(shape=21, alpha=0.7, size=2, stroke=NA) +
  geom_point(data=represt_summary %>% filter(alpha == 1),
             aes(x=elect_system, y=inv_dist), size=5,
             shape=21) +
  geom_errorbar(data=represt_summary %>% filter(alpha == 1),
                aes(ymin=inv_dist-(2*se_reprs),
                    ymax=inv_dist+(2*se_reprs)),
                width=0.1) +
  scale_x_discrete("Electoral System", labels = c("First-Past-The-Post", "Proportional\nRepresentation")) +
  scale_y_continuous("Representativeness") +
  # scale_color_discrete(labels = c("FPTP", "PR")) +
  scale_fill_manual(values=c("#F8766D", "#619CFF"),
                    labels=c("FPTP", "PR")) +
  guides(fill = guide_legend(title = "Electoral\nSystem")) +
  theme_classic() +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_text(size=13),
        axis.text=element_text(size=13),
        legend.position = "None")

biased_reprst_plot <- represt_df %>% 
  filter(alpha == 1.5) %>% 
  ggplot(aes(x=elect_system, y=inv_dist, fill=elect_system)) +
  geom_point(shape=21, alpha=0.7, size=2, stroke=NA) +
  geom_point(data=represt_summary %>% filter(alpha == 1.5),
             aes(x=elect_system, y=inv_dist), size=5,
             shape=21) +
  geom_errorbar(data=represt_summary %>% filter(alpha == 1.5),
                aes(ymin=inv_dist-(2*se_reprs),
                    ymax=inv_dist+(2*se_reprs)),
                width=0.1) +
  scale_x_discrete("Electoral System", labels = c("First-Past-The-Post", "Proportional\nRepresentation")) +
  scale_y_continuous("Representativeness") +
  # scale_color_discrete(labels = c("FPTP", "PR")) +
  scale_fill_manual(values=c("#F8766D", "#619CFF"),
                    labels=c("FPTP", "PR")) +
  guides(fill = guide_legend(title = "Electoral\nSystem")) +
  theme_classic() +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_text(size=13),
        axis.text=element_text(size=13),
        legend.position = "None")


# Difference in representativeness between fptp and pr as bias increases
elect_system_list <- c('one_per_party', 'proportional_rep')

biased_represt_df <- NULL

for (elect_system in elect_system_list) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('elected_biased_representation_', elect_system, '.csv'))
  
  biased_represt_results <- read_csv(result_path) %>%
    mutate(elect_system = elect_system,
           inv_dist = 1 - (avg_dist/2))
  
  biased_represt_df <- rbind(biased_represt_df, biased_represt_results)
}

biased_represt_df %>% 
  group_by(elect_system, a) %>% 
  summarize(representation.avg = mean(inv_dist),
            dist.avg = mean(avg_dist))

biased_fptp_pr <- biased_represt_df %>% 
  group_by(elect_system, a) %>% 
  summarize(representation.avg = mean(inv_dist)) %>% 
  pivot_wider(id_cols = a, names_from = elect_system, values_from = representation.avg) %>% 
  mutate(diff_rep = proportional_rep - one_per_party)
  
biased_diff_plot <- biased_fptp_pr %>% 
  ggplot(aes(x=a, y=diff_rep)) +
  geom_line() +
  scale_x_continuous(expression(paste("Public Ideological Bias"))) +
  scale_y_continuous(expression(atop(Delta*"Representativeness", "(PR - FPTP)"))) +
  theme_classic() +
  theme(axis.title=element_text(size=16),
        axis.title.y=element_text(size=13),
        axis.text=element_text(size=16))


# composing the figure

balance_title <- ggdraw() + 
  draw_label(
    "Balanced Population",
    fontface = 'bold',
    size = 14,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

bias_title <- ggdraw() + 
  draw_label(
    "Biased Population",
    fontface = 'bold',
    size = 14,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

letters <- c("a", "b", "c", "d", "e", "f")
label_list <- list()
for (i in 1:6) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0.01, y = 0.1,
      hjust = 0, vjust = 0
    )
}

top_panel <- plot_grid(label_list[[1]], NA, label_list[[2]],
                       balanced_reps_plot, NA, balanced_reprst_plot,
                       ncol = 3, rel_heights = c(0.12, 1),
                       rel_widths = c(1, 0.1, 1))

middle_panel <- plot_grid(label_list[[3]], NA, label_list[[4]],
                          biased_reps_plot, NA, biased_reprst_plot,
                          ncol = 3, rel_heights = c(0.12, 1),
                          rel_widths = c(1, 0.1, 1))

fig1 <- plot_grid(tmp_legend, NA, balance_title, top_panel,
                  bias_title, middle_panel,
                  label_list[[5]], biased_diff_plot,
                  ncol = 1, rel_heights = c(0.08, 0.01, 0.1, 1,
                                            0.1, 1, 0.12, 1))

ggsave(file.path(ce_result_path, paste0("election_fig1.jpg")),
       plot = fig1, height = 8, width = 12)











# Figure 2: Emergence of party and representation

# Party initialization plots
party_sd <- 0.1
position_range <- seq(-1, 1, by=0.01)

gaussian <- function(x, p_center, p_sd) {
  return( exp(-((x - p_center)^2) / (2*p_sd^2))  )
}

# moderate party emergence (-0.66, 0*, 0.66)
party_positions <- c(-0.66, 0.66, 0)
partym_range_list <- list()

for (i in 1:3) {
  ps <- party_positions[i]
  pdf <- gaussian(position_range, ps, party_sd)
  partym_range_list[[i]] <- data.frame(x=position_range, pdf=pdf,
                                       party=i, center=ps)
}

party_mod_range <- rbind(partym_range_list[[1]], partym_range_list[[2]], partym_range_list[[3]])
party_mod_range <- party_mod_range %>% 
  filter(x < center+(4*party_sd) & x > center-(4*party_sd)) %>% 
  mutate(party_group = ifelse(party < 3, "old", "new"),
         party_group = fct_relevel(party_group,c("old","new")))

mod_range_plot <- party_mod_range %>% 
  ggplot(aes(x=x, y=pdf, color=party_group, group=party)) +
  geom_line() +
  scale_y_continuous(expand = c(0,0)) +
  scale_x_continuous("Ideology", limits = c(-1,1)) +
  scale_color_manual(labels=c("Old Party", "New Party"),
                     values=c("#999999", "#E69F00")) +
  theme_classic() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_text(size=14),
        axis.text.x=element_text(size=12),
        legend.direction="horizontal",
        legend.text=element_text(size=14),
        legend.title=element_blank(),
        legend.position="top")


# illustrative case of party emergence

elect_system_list <- c('one_per_party', 'proportional_rep')
mod_illst_plots <- list()

for (i in 1:length(elect_system_list)) {
  elected_path <- file.path(box_path, 'ComplexElection', 'results',
                            paste0('elected_pref_', elect_system_list[i], '_moderate.csv'))
  
  elected_reps <- read_csv(elected_path) %>% 
    mutate(party_group = ifelse(party < 2, "old", "new"),
           party_group = fct_relevel(party_group,c("old","new")))
  
  res_v_rep_hist <- ggplot(elected_reps, aes(x=pos, fill=party_group)) +
    # geom_histogram(aes(y=..count../sum(..count..), fill="resident"),
    #                bins=50, color="black", alpha=0.5) +
    geom_histogram(aes(y=..count../sum(..count..)),
                   bins=50, color="black", alpha=0.5) +
    scale_x_continuous("Ideology", limits = c(-1,1)) +
    scale_y_continuous(expand = c(0,0)) +
    labs(y = "Proportion", fill = NULL) +
    scale_fill_manual(labels = c("Old Party", "New Party"),
                      values = c("#999999", "#E69F00")) +
    theme_classic() +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position="None")
  
  mod_illst_plots[[i]] <- res_v_rep_hist
}


# Result 1
# New parties are suppressed when voters are more strategic and rely on history especially under FPTP

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 2
pop_mag <- 5
op_distr <- 'uniform'
sim_type <- 'when_pemerge'

elect_system_list <- c('one_per_party', 'proportional_rep')
heatmap_list <- list()

party_num <- 2

for (i in 1:length(elect_system_list)) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system_list[i],
                                  '_p', party_num, '_', 'moderate',
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


fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

heatmaps <- plot_grid(fptp_title, prop_title,
                      heatmap_list[[1]], heatmap_list[[2]],
                      ncol = 2, labels = c("b", "c"), label_size = 20,
                      rel_widths = c(1, 1, 0.2), rel_heights = c(0.1, 1))

# enep_heatmap_grid <- plot_grid(heatmaps, heatmap_legend, ncol = 2,
#                                rel_widths = c(1, 0.15))




# Result 2
# Geographical homophilic sorting under fptp
batch_id <- 2
pop_mag <- 5
party_num <- 2
sim_type <- 'geo_sort'
elect_system_list <- c('one_per_party', 'proportional_rep')
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
  isort_agg_result <- step_results %>%
    filter(step > 5, beta == 0.5) %>% 
    group_by(sim_id, ideo_sort, alpha) %>% 
    summarize(lt_enep=mean(lt_enep))
  
  # New palette: values = c("#FFEDBB", "#FEB24C", "#FF3F21")
  # Viridis: values = c("#7ad151", "#2a788e", "#440154")
  panel_ind <- i
  isort_plot_list[[panel_ind]] <- isort_agg_result %>% 
    ggplot(aes(x=ideo_sort, y=lt_enep, color=as.factor(alpha))) +
    geom_point(position = position_jitter(w = 0.02, h = 0), alpha=0.7) +
    scale_x_continuous(expression(paste("Homophily (", h, ")")), expand=c(0.02,0.02)) +
    scale_y_continuous("ENEP", limits=c(1,3.1)) +
    scale_color_manual(values = c("#7ad151", "#2a788e", "#440154"),
                       labels = c(0.1, 0.5, 0.9)) +
    guides(color=guide_legend(title=expression(paste("Strategic\nTendency (", alpha, ")")))) +
    theme_classic() +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          legend.title = element_text(size=14),
          legend.text = element_text(size=12))
}

# agg_sort_results <- rbind(agg_result_list[[1]], agg_result_list[[2]])

# extract legend and remove it from the plot
isort_legend <- get_legend(isort_plot_list[[1]])

isort_plot_list[[1]] <- isort_plot_list[[1]] + theme(legend.position = "none")
isort_plot_list[[2]] <- isort_plot_list[[2]] + theme(legend.position = "none")

# enep_sort_pg <- plot_grid(isort_plot_list[[1]], isort_plot_list[[2]],
#                           isort_legend, ncol = 3, rel_widths = c(1,1,0.15))



# Result 3
# Representation and emergence of party

batch_id <- 2
pop_mag <- 5
op_distr <- 'uniform'
party_num <- 2
new_party <- 'moderate'
sim_type <- 'when_pemerge'

votep_cols <- paste0('vote_prop', 0:party_num)
seatp_cols <- paste0('seat_prop', 0:party_num)

elect_system_list <- c('one_per_party', 'proportional_rep')

represt_df <- NULL

for(elect_system in elect_system_list) {
  result_path <- file.path(box_path, 'ComplexElection', 'results',
                           paste0('estep', pop_mag, 'k_', elect_system,
                                  '_p', party_num, '_', new_party,
                                  '_', sim_type, batch_id, '.csv'))
  
  step_results <- read_csv(result_path) %>%
    select(-district_num, -rep_num)
  
  # fill na for vote props and seat props when parties don't get any vote (or they didn't exist)
  elect_results <- step_results[c(votep_cols, seatp_cols)]
  elect_results[is.na(elect_results)] <- 0
  step_results[c(votep_cols, seatp_cols)] <- elect_results
  
  step_results$lt_enep <- apply(step_results[votep_cols], 1, lt_enep)
  
  # calculate representativeness for each simulation with a given set of parameters (alpha, beta)
  # average distant range [0, 2]
  represt_tmp <- step_results %>% 
    filter(step > 5, alpha < 1) %>% 
    group_by(sim_id, alpha, beta) %>% 
    summarize(lt_enep = mean(lt_enep), avg_close_dist = mean(avg_close_elected)) %>% 
    mutate(inv_dist = 1 - (avg_close_dist/2))
  represt_tmp$elect_system <- elect_system
  
  represt_df <- rbind(represt_df, represt_tmp)
}

# summarize representation data to make dot and whisker in the plot
represt_summary <- represt_df %>% 
  group_by(elect_system) %>% 
  summarize(avg_reprs = mean(inv_dist), sd_reprs = sd(inv_dist),
            avg_lt_enep = mean(lt_enep), sd_lt_enep = sd(lt_enep),
            data_count = n()) %>% 
  mutate(se_reprs = sd_reprs/sqrt(data_count), se_lt_enep = sd_lt_enep/sqrt(data_count))


represt_plot <- represt_df %>% 
  ggplot(aes(x=lt_enep, y=inv_dist, fill=elect_system)) +
  geom_point(shape=21, alpha=0.3, size=1.5, stroke=NA) +
  geom_point(data=represt_summary, aes(x=avg_lt_enep, y=avg_reprs), size=5,
             shape=21, color="black") +
  scale_x_continuous("Effective Number of Electoral Parties (ENEP)") +
  scale_y_continuous("Representativeness") +
  scale_fill_manual(values=c("#F8766D", "#619CFF"),
                    labels=c("FPTP", "PR")) +
  # guides(fill = guide_legend(title = "Electoral\nSystem")) +
  theme_classic() +
  theme(axis.title=element_text(size=14),
        axis.text=element_text(size=12),
        legend.title = element_blank(),
        legend.text = element_text(size=14))

# extract legend and remove it from the plot
represt_legend <- get_legend(represt_plot)
represt_plot <- represt_plot + theme(legend.position = "none")

# represt_pg <- plot_grid(represt_plot, NA,
#                         ncol = 2, rel_widths = c(2, 0.3),
#                         label_size = 18, labels = c("a", NA))

# ggsave(file.path(ce_result_path, paste0("enep_rep_distance.jpeg")),
#        plot = represt_plot, height = 6, width = 12)

# labels
letters <- c("a", "b", "c", "d", "e", "f", "g", "h")
label_list <- list()
for (i in 1:8) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0.01, y = 0.15,
      hjust = 0, vjust = 0
    )
}


# combining 3 panels into one figure
figure2 <- plot_grid(label_list[[1]], label_list[[2]], NA,
                     mod_range_plot, represt_plot, represt_legend,
                     NA, NA, NA,
                     fptp_title, prop_title, NA,
                     label_list[[3]], label_list[[4]], NA,
                     mod_illst_plots[[1]], mod_illst_plots[[2]], NA,
                     label_list[[5]], label_list[[6]], NA,
                     heatmap_list[[1]], heatmap_list[[2]], heatmap_legend,
                     label_list[[7]], label_list[[8]], NA,
                     isort_plot_list[[1]], isort_plot_list[[2]], isort_legend,
                     ncol = 3, rel_widths = c(1, 1, 0.25),
                     rel_heights = c(0.1, 1, 0.03, 0.12, 0.1, 0.7, 0.1, 1, 0.12, 1))

# label_size = 16,
# labels = c(NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA,
#            NA, NA, NA)

# plot_grid(represt_pg, NA, bottom_panel,
#           ncol = 1, rel_heights = c(0.5, 0.03, 1))

ggsave(file.path(ce_result_path, "election_fig2_2.jpg"),
       plot = figure2,
       height = 10, width = 10)








# Figure 3: Extremist party emergence

letters <- c("a", "b", "c", "d", "e")
label_list <- list()
for (i in 1:5) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 16,
      x = 0, y = 0.1,
      hjust = 0, vjust = 0
    )
}

# Result 4
# Extremist party is suppressed more under FPTP than PR (alpha=0.5, beta=0.6)

# Party initialization plots
party_sd <- 0.1
position_range <- seq(-1, 1, by=0.01)

gaussian <- function(x, p_center, p_sd) {
  return( exp(-((x - p_center)^2) / (2*p_sd^2))  )
}

# extremist party emergence (-0.25, 0.25, 0.75*)
party_positions <- c(-0.25, 0.25, 0.75)
partye_range_list <- list()

for (i in 1:3) {
  ps <- party_positions[i]
  pdf <- gaussian(position_range, ps, party_sd)
  partye_range_list[[i]] <- data.frame(x=position_range, pdf=pdf,
                                      party=i, center=ps)
}

party_ext_range <- rbind(partye_range_list[[1]], partye_range_list[[2]], partye_range_list[[3]])
party_ext_range <- party_ext_range %>% 
  filter(x < center+(4*party_sd) & x > center-(4*party_sd)) %>% 
  mutate(party_group = ifelse(party < 3, "old", "new"),
         party_group = fct_relevel(party_group,c("old","new")))

ext_range_plot <- party_ext_range %>% 
  ggplot(aes(x=x, y=pdf, color=party_group, group=party)) +
  geom_line() +
  scale_y_continuous(expand = c(0,0)) +
  scale_x_continuous("Ideology", limits = c(-1,1)) +
  scale_color_manual(labels=c("Old Party", "New Party"),
                     values=c("#999999", "#E69F00")) +
  theme_classic() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_text(size=14),
        axis.text.x=element_text(size=12),
        legend.text=element_text(size=14),
        legend.title=element_blank())


# illustrative case of party emergence

elect_system_list <- c('one_per_party', 'proportional_rep')
ext_illst_plots <- list()

for (i in 1:length(elect_system_list)) {
  elected_path <- file.path(box_path, 'ComplexElection', 'results',
                            paste0('elected_pref_', elect_system_list[i], '_extreme.csv'))
  
  elected_reps <- read_csv(elected_path) %>% 
    mutate(party_group = ifelse(party < 2, "old", "new"),
           party_group = fct_relevel(party_group,c("old","new")))
  
  res_v_rep_hist <- ggplot(elected_reps, aes(x=pos, fill=party_group)) +
    # geom_histogram(aes(y=..count../sum(..count..), fill="resident"),
    #                bins=50, color="black", alpha=0.5) +
    geom_histogram(aes(y=..count../sum(..count..)),
                   bins=50, color="black", alpha=0.5) +
    scale_x_continuous("Ideology", limits = c(-1,1)) +
    scale_y_continuous(expand = c(0,0)) +
    labs(y = "Proportion", fill = NULL) +
    scale_fill_manual(labels = c("Old Party", "New Party"),
                      values = c("#999999", "#E69F00")) +
    theme_classic() +
    theme(axis.title=element_text(size=14),
          axis.text=element_text(size=12),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position="None")
  
  ext_illst_plots[[i]] <- res_v_rep_hist
}


# Emergence of extremist party across parameters

party_num <- 3
votep_cols <- paste0('vote_prop', 0:(party_num-1))
seatp_cols <- paste0('seat_prop', 0:(party_num-1))

batch_id <- 2
pop_mag <- 5
op_distr <- 'uniform'
sim_type <- 'when_pemerge'

elect_system_list <- c('one_per_party', 'proportional_rep')
ext_heatmap_list <- list()

party_num <- 2

for (i in 1:length(elect_system_list)) {
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
  ext_heatmap_list[[panel_ind]] <- enep_heatmap
}

# extract shared legend and remove legends from both plots
heatmap_legend <- get_legend(ext_heatmap_list[[1]])

ext_heatmap_list[[1]] <- ext_heatmap_list[[1]] + theme(legend.position = "none")
ext_heatmap_list[[2]] <- ext_heatmap_list[[2]] + theme(legend.position = "none")


fptp_title <- ggdraw() + 
  draw_label(
    "First-Past-The-Post",
    fontface = 'bold',
    size = 16,
    x = 0.6,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Representation",
    fontface = 'bold',
    size = 16,
    x = 0.6,
    vjust = 0.5,
    angle = 0
  )

# top panel plot of illustrative condition
ext_legend <- get_legend(ext_range_plot)
ext_range_plot <- ext_range_plot + theme(legend.position = "none")

ext_range_panel <- plot_grid(label_list[[1]], NA,
                             ext_range_plot, ext_legend,
                             ncol = 2, rel_widths = c(1, 0.15),
                             rel_heights = c(0.1, 1))

ext_bottom_panel <- plot_grid(fptp_title, prop_title, NA,
                              label_list[[2]], label_list[[3]], NA,
                              ext_illst_plots[[1]], ext_illst_plots[[2]], NA,
                              label_list[[4]], label_list[[5]], NA,
                              ext_heatmap_list[[1]], ext_heatmap_list[[2]], heatmap_legend,
                              ncol = 3, rel_widths = c(1, 1, 0.3),
                              rel_heights = c(0.2, 0.1, 1, 0.1, 1))

# pemerge_legend <- plot_grid(NA, shared_rep_legend,
#                             ncol = 2, rel_widths = c(0.2, 2))

figure3 <- plot_grid(ext_range_panel, ext_bottom_panel, ncol=1, rel_heights = c(1, 2.4))
ggsave(file.path(ce_result_path, paste0("election_fig3_2.jpg")),
       plot = figure3, height = 10, width = 12)










# Figure 4: Party Radicalization and Entrenchment

# Party initialization plots
party_sd <- 0.1
position_range <- seq(-1, 1, by=0.01)

gaussian <- function(x, p_center, p_sd) {
  return( exp(-((x - p_center)^2) / (2*p_sd^2))  )
}

# Gradual Radicalization illustration
party_positions <- c(-0.5, 0)
partye_range_list <- list()

for (i in 1:2) {
  ps <- party_positions[i]
  pdf <- gaussian(position_range, ps, party_sd)
  partye_range_list[[i]] <- data.frame(x=position_range, pdf=pdf,
                                       party=i, center=ps)
}

party_ext_range <- rbind(partye_range_list[[1]], partye_range_list[[2]])
party_ext_range <- party_ext_range %>% 
  filter(x < center+(4*party_sd) & x > center-(4*party_sd)) %>% 
  mutate(party_group = ifelse(party == 2, "old", "new"),
         party_group = fct_relevel(party_group,c("old","new")))

init_radical_plot <- party_ext_range %>% 
  ggplot(aes(x=x, y=pdf, group=party)) +
  geom_line() +
  geom_point(data=tibble(x=party_positions, pdf=0.5, party=c(1,2)), size=5) +
  geom_vline(xintercept = party_positions, linetype = "dashed") +
  annotate("segment", y=0.5,yend=0.5,x=0.07,xend=0.7,arrow=arrow(type = "closed"), linewidth=1) +
  annotate("text", y=0.65,x=0.6,label="Gradual\nRadicalization", size=5) +
  scale_y_continuous(expand = c(0,0)) +
  scale_x_continuous("Ideology", limits = c(-1,1)) +
  # scale_color_manual(values=c("#999999", "#E69F00")) +
  theme_classic() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_text(size=16),
        axis.text.x=element_text(size=14),
        legend.position="none")


# Result 5
# Radicalization

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

# Creating (multiple) plots
# Ideology over time

ideo_change <- all_rad_results %>% 
  distinct(mov_ppos, step) %>% 
  ggplot(aes(x=step, y=mov_ppos)) + 
  geom_line() +
  annotate("segment", y=0,yend=0.5,x=12.5,xend=12.5, linewidth=1, linetype="dashed") +
  annotate("segment", y=0.5,yend=0.5,x=0,xend=12.5, linewidth=1, linetype="dashed") +
  annotate("text", y=0.65,x=6.5, label="Radicalization\nThreshold", size=5) +
  scale_x_continuous("Electoral Cycle") +
  scale_y_continuous("Party Ideology") +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14))


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
        legend.position = "None")

ggsave(file.path(ce_result_path, "election_presentation_rad_fptp3.jpg"),
       plot = fptp_voteshare,
       height = 4, width = 4)


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


# Vote share deviation from sincere voting over time
# FPTP
fptp_votedev <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'one_per_party') %>% 
  ggplot(aes(x=step, y=share_deviation, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=fptp_avg_segments,
               aes(x=step_start, y=share_deviation_start,
                   xend=step_end, yend=share_deviation_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=fptp_arrow_segments,
               aes(x=step_start, y=share_deviation_start,
                   xend=step_end, yend=share_deviation_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste(s[actual] - s[sincere])), limits=c(-0.4, 0.6)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_discrete(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)),
                                  expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)),
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)))) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.position = "None")

# PR
pr_votedev <- agg_rad_results %>% 
  filter(alpha %in% c(0.3, 0.6),
         beta %in% c(0.3, 0.6),
         elect_system == 'proportional_rep') %>% 
  ggplot(aes(x=step, y=share_deviation, color=interaction(alpha, beta), group=interaction(sim_id, alpha, beta))) +
  geom_line(linewidth=0.5, alpha=0.3) +
  geom_segment(data=pr_avg_segments,
               aes(x=step_start, y=share_deviation_start,
                   xend=step_end, yend=share_deviation_end,
                   group=interaction(alpha, beta)), linewidth=1) +
  geom_segment(data=pr_arrow_segments,
               aes(x=step_start, y=share_deviation_start,
                   xend=step_end, yend=share_deviation_end,
                   group=interaction(alpha, beta)),
               linewidth=1, arrow = arrow()) +
  geom_vline(xintercept = 12.5, linetype="dashed", linewidth=1) +
  scale_x_continuous("Electoral Cycle", expand=c(0.02,0.02)) +
  scale_y_continuous(expression(paste(s[actual] - s[sincere])), limits=c(-0.4, 0.6)) +
  guides(color=guide_legend(title="Parameters", title.position="top")) +
  scale_color_discrete(labels = c(expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.3)), 
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.3)),
                                  expression(paste(alpha, "=", 0.3, ",", beta, "=", 0.6)),
                                  expression(paste(alpha, "=", 0.6, ",", beta, "=", 0.6)))) +
  theme_classic() +
  theme(axis.title=element_text(size=18),
        axis.text=element_text(size=14),
        legend.title = element_text(size=16),
        legend.text = element_text(size=14))

# extract shared legend and remove legends from both plots
shared_rad_legend <- get_legend(pr_votedev)
pr_votedev <- pr_votedev + theme(legend.position = "none")

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

letters <- c("a", "b", "c", "d", "e", "f")
label_list <- list()
for (i in 1:6) {
  label_list[[i]] <- ggdraw() + 
    draw_label(
      letters[i],
      fontface = 'bold',
      size = 18,
      x = 0, y = 0.1,
      hjust = 0, vjust = 0
    )
}

main_rad_panel <- plot_grid(NA, NA, fptp_title, prop_title,
                            label_list[[1]], NA, label_list[[3]], label_list[[4]],
                            init_radical_plot, NA, fptp_voteshare, pr_voteshare,
                            label_list[[2]], NA, label_list[[5]], label_list[[6]],
                            ideo_change, NA, fptp_votedev, pr_votedev,
                            ncol = 4, rel_widths = c(1, 0.1, 1, 1),
                            rel_heights = c(0.15, 0.1, 1, 0.1, 1))

plot_grid(main_rad_panel, NA, shared_rad_legend,
          ncol = 3, rel_widths = c(1, 0.01, 0.15))

ggsave(file.path(ce_result_path, "election_fig4.jpg"),
       height = 8, width = 12)


# rad_grid <- plot_grid(shared_rad_legend, label_list[[4]], rad_plot_list[[1]],
#                       label_list[[5]], rad_plot_list[[2]], ncol = 1,
#                       rel_heights = c(0.6, 0.07, 1, 0.07, 1))

# plot_grid(rep_grid, NA, rad_grid,
#           pemerge_legend, NA, NA,
#           ncol = 3, rel_widths = c(1, 0.1, 1), rel_heights = c(1,0.1))

# ggsave(file.path(ce_result_path, paste0("party_radicalization1.jpeg")),
#        plot = rad_grid, height = 6, width = 12)

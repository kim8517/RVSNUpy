# if temp_type==1:
#     if fit_lines.stddev.value*2*np.sqrt(2*np.log(2))>resolution:
#         if fit_lines.amplitude.value>0:
#             self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
#             self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
#         else:
#             self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
# elif temp_type==2:
#     if fit_lines.stddev.value*2*np.sqrt(2*np.log(2))>resolution:
#         if fit_lines.amplitude.value<0:
#             self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
#             self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
#         else:
#             self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
# elif temp_type==0:
#     if fit_lines.stddev.value*2*np.sqrt(2*np.log(2))>resolution:
#         self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
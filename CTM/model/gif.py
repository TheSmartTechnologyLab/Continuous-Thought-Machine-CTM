def make_gif(predictions, certainties, targets, pre_activations, post_activations, attention, inputs_to_model, filename):
    def reshape_attention_weights(attention_weights):
        T, B = attention_weights.shape[0], attention_weights.shape[1]
        grid_size = math.sqrt(attention_weights.shape[-1])
        assert grid_size.is_integer(), f'Grid size should be a perfect square, but got {attention_weights.shape[-1]}'
        H_ATTENTION = W_ATTENTION = int(grid_size)
        attn_weights_reshaped = attention_weights.reshape(T, B, -1, H_ATTENTION, W_ATTENTION)
        return attn_weights_reshaped.mean(2)

    batch_index = 0
    n_neurons_to_visualise = 16
    figscale = 0.28
    n_steps = len(pre_activations)
    heCTMap_cmap = sns.color_palette("viridis", as_cmap=True)
    frames = []

    attention = reshape_attention_weights(attention)

    these_pre_acts = pre_activations[:, batch_index, :]
    these_post_acts = post_activations[:, batch_index, :]
    these_inputs = inputs_to_model[batch_index,:, :, :]
    these_attention_weights = attention[:, batch_index, :, :]
    these_predictions = predictions[batch_index, :, :]
    these_certainties = certainties[batch_index, :, :]
    this_target = targets[batch_index]

    class_labels = [str(i) for i in range(these_predictions.shape[0])]

    mosaic = [['img_data', 'img_data', 'attention', 'attention', 'probs', 'probs', 'probs', 'probs'] for _ in range(2)] + \
             [['img_data', 'img_data', 'attention', 'attention', 'probs', 'probs', 'probs', 'probs'] for _ in range(2)] + \
             [['certainty'] * 8] + \
             [[f'trace_{ti}'] * 8 for ti in range(n_neurons_to_visualise)]

    for stepi in range(n_steps):
        fig_gif, axes_gif = plt.subplot_mosaic(mosaic=mosaic, figsize=(31*figscale*8/4, 76*figscale))
        probs = softmax(these_predictions[:, stepi])
        colors = [('g' if i == this_target else 'b') for i in range(len(probs))]

        axes_gif['probs'].bar(np.arange(len(probs)), probs, color=colors, width=0.9, alpha=0.5)
        axes_gif['probs'].set_title('Probabilities')
        axes_gif['probs'].set_xticks(np.arange(len(probs)))
        axes_gif['probs'].set_xticklabels(class_labels, fontsize=24)
        axes_gif['probs'].set_yticks([])
        axes_gif['probs'].tick_params(left=False, bottom=False)
        axes_gif['probs'].set_ylim([0, 1])
        for spine in axes_gif['probs'].spines.values():
            spine.set_visible(False)
        axes_gif['probs'].tick_params(left=False, bottom=False)
        axes_gif['probs'].spines['top'].set_visible(False)
        axes_gif['probs'].spines['right'].set_visible(False)
        axes_gif['probs'].spines['left'].set_visible(False)
        axes_gif['probs'].spines['bottom'].set_visible(False)

        # Certainty
        axes_gif['certainty'].plot(np.arange(n_steps), these_certainties[1], 'k-', linewidth=2)
        axes_gif['certainty'].set_xlim([0, n_steps-1])
        axes_gif['certainty'].axvline(x=stepi, color='black', linewidth=1, alpha=0.5)
        axes_gif['certainty'].set_xticklabels([])
        axes_gif['certainty'].set_yticklabels([])
        axes_gif['certainty'].grid(False)
        for spine in axes_gif['certainty'].spines.values():
            spine.set_visible(False)

        # Neuron Traces
        for neuroni in range(n_neurons_to_visualise):
            ax = axes_gif[f'trace_{neuroni}']
            pre_activation = these_pre_acts[:, neuroni]
            post_activation = these_post_acts[:, neuroni]
            ax_pre = ax.twinx()

            ax_pre.plot(np.arange(n_steps), pre_activation, color='grey', linestyle='--', linewidth=1, alpha=0.4)
            color = 'blue' if neuroni % 2 else 'red'
            ax.plot(np.arange(n_steps), post_activation, color=color, linewidth=2, alpha=1.0)

            ax.set_xlim([0, n_steps-1])
            ax_pre.set_xlim([0, n_steps-1])
            ax.set_ylim([np.min(post_activation), np.max(post_activation)])
            ax_pre.set_ylim([np.min(pre_activation), np.max(pre_activation)])

            ax.axvline(x=stepi, color='black', linewidth=1, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)
            ax_pre.set_xticklabels([])
            ax_pre.set_yticklabels([])
            ax_pre.grid(False)

            for spine in ax.spines.values():
                spine.set_visible(False)
            for spine in ax_pre.spines.values():
                spine.set_visible(False)

        # Input image
        this_image = these_inputs[0]
        this_image = (this_image - this_image.min()) / (this_image.max() - this_image.min() + 1e-8)
        axes_gif['img_data'].set_title('Input Image')
        axes_gif['img_data'].imshow(this_image, cmap='binary', vmin=0, vmax=1)
        axes_gif['img_data'].axis('off')

        # Attention
        this_input_gate = these_attention_weights[stepi]
        gate_min, gate_max = np.nanmin(this_input_gate), np.nanmax(this_input_gate)
        if not np.isclose(gate_min, gate_max):
            normalized_gate = (this_input_gate - gate_min) / (gate_max - gate_min + 1e-8)
        else:
            normalized_gate = np.zeros_like(this_input_gate)
        attention_weights_heCTMap = heCTMap_cmap(normalized_gate)[:,:,:3]

        axes_gif['attention'].imshow(attention_weights_heCTMap, vmin=0, vmax=1)
        axes_gif['attention'].axis('off')
        axes_gif['attention'].set_title('Attention')

        fig_gif.tight_layout()
        canvas = fig_gif.canvas
        canvas.draw()
        image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:, :, :3]
        frames.append(image_numpy)
        plt.close(fig_gif)


    mediapy.show_video(frames, width=400, codec="gif")
    imageio.mimsave(filename, frames, fps=5, loop=100)